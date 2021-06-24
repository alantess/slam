import numpy as np
import cv2 as cv
from sklearn.preprocessing import MinMaxScaler
from slam.extract.normalize import *


class FeatureExtractor(object):
    def __init__(self, H, W):
        self.H = H
        self.W = W
        self.scaler = MinMaxScaler()
        self.bf = cv.BFMatcher(cv.NORM_HAMMING)
        self.orb = cv.ORB_create()
        # Camera intrinsics and extrinsics
        self.P = self._calibrate()

        # Camera intrinsics and extrinsics
        self.c_u = self.P[0, 2]
        self.c_v = self.P[1, 2]
        self.f_u = self.P[0, 0]
        self.f_v = self.P[1, 1]
        # self.b_x = self.P[0, 3] / (-self.f_u)  # relative
        # self.b_y = self.P[1, 3] / (-self.f_v)
        self.kps = None
        self.kps_matches = None
        self.kps2_matches = None
        self.last = None
        self.img = None
        self.xyz = [0, 0, 0]

    def _calibrate(self):
        res = np.array([[3000, 0, self.W / 2], [0, 3000, self.H / 2],
                        [0, 0, 1]])
        return res

    # Extracts features
    def extract(self, img):
        proj = np.array([0, 0, 0])
        pts1, pts2, kpts, matches = self.extract_feats(img)
        points1 = self.cart2hom(pts1)
        points2 = self.cart2hom(pts2)
        tripoints = []
        if points1.ndim != 1 or points2.ndim != 1:
            points1_norm = np.dot(np.linalg.inv(self.P), points1)
            points2_norm = np.dot(np.linalg.inv(self.P), points2)
            E = compute_essential_normalized(points1_norm, points2_norm)
            P1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
            P2s = compute_P_from_essential(E)
            ind = -1
            for i, P2 in enumerate(P2s):
                d1 = reconstruct_one_point(points1_norm[:, 0],
                                           points2_norm[:, 0], P1, P2)

                P2_homogenous = np.linalg.inv(np.vstack([P2, [0, 0, 0, 1]]))

                d2 = np.dot(P2_homogenous[:3, :4], d1)

                if d1[2] > 0 and d2[2] > 0:
                    ind = i

            P2 = np.linalg.inv(np.vstack([P2s[ind], [0, 0, 0, 1]]))[:3, :4]
            tripoints = triangulation(points1_norm, points2_norm, P1, P2)
        else:
            print('BAD ARRAY')

        if len(tripoints) > 0:
            x = [pt for pt in tripoints[0]]
            y = [-pt for pt in tripoints[1]]
            z = [-pt for pt in tripoints[2]]

            for i in range(tripoints.shape[1]):
                cur = np.array([x[i], y[i], z[i]])
                proj = np.vstack((proj, cur))

            proj = proj[1:, :]

        return self.img, proj

    def extract_feats(self, img):
        ret = []
        # Extracts ORB
        feats = cv.goodFeaturesToTrack(
            np.mean(img, axis=2).astype(np.uint8), 4000, 0.02, 3)
        kps = [cv.KeyPoint(x=f[0][0], y=f[0][1], _size=30) for f in feats]

        self.img = cv.drawKeypoints(img,
                                    kps,
                                    None,
                                    color=(248, 186, 255),
                                    flags=0)
        kpts, des = self.orb.compute(img, kps)
        if self.last is not None:
            matches = self.bf.knnMatch(des, self.last['des'], 2)
            for m, n in matches:
                if m.distance < 0.55 * n.distance:
                    if m.distance < 64:
                        kpt1_match = kpts[m.queryIdx]
                        kpt2_match = self.last["kpts"][m.trainIdx]
                        ret.append((kpt1_match, kpt2_match))
            coords_1 = np.asarray([kpts[m.queryIdx].pt for m, n in matches])
            coords_2 = np.asarray(
                [self.last["kpts"][m.trainIdx].pt for m, n in matches])

            retval, mask = cv.findHomography(coords_1, coords_2, cv.RANSAC,
                                             100.0)

            mask = mask.ravel()
            pts1 = coords_1[mask == 1]
            pts2 = coords_2[mask == 1]
            self.last = {"kpts": kpts, "des": des}
            return pts1.T, pts2.T, kpts, ret
        else:
            self.last = {"kpts": kpts, "des": des}
            return np.array([0]), np.array([0]), 0, 0

    def get_direct_lines(self, img):
        lsd = cv.ximgproc.createFastLineDetector()
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        lines = lsd.detect(img)
        img = lsd.drawSegments(img, lines)
        return img, lines

    def cart2hom(self, array):
        """Convert array from Cartesian -> Homogenous (2 dimensions -> 3 dimensions)"""
        if array.ndim == 1:
            return np.array([0])

        else:
            array_3dim = np.asarray(np.vstack([array,
                                               np.ones(array.shape[1])]))
            return array_3dim

    def generated_3d_lines(self, lines):
        # reshape to UxV
        lines = lines.reshape(2, -1)

        lines = lines.T
        u = lines[:, 0]
        v = lines[:, 1]
        min_x = np.min(lines[:, 0])
        min_y = np.min(lines[:, 1])
        margin = 0.75

        points = np.ones((lines.shape[0], 3))
        points[:, 0] = u
        points[:, 1] = v

        points = self.lidar_generation(points)
        return points

    def lidar_generation(self, pts):
        uv_depth = self._2d_to_3d(pts)
        # uv_depth = self.scaler.fit_transform(uv_depth)
        n = uv_depth.shape[0]
        z = uv_depth[:, 2] * 0.2
        x = ((uv_depth[:, 0] - self.c_u) * z) / self.f_u
        y = ((uv_depth[:, 1] - self.c_v) * z) / self.f_v

        pts_3d_rect = np.zeros((n, 3))
        pts_3d_rect[:, 0] = x
        pts_3d_rect[:, 1] = y
        pts_3d_rect[:, 2] = z

        pts_3d_rect = self.scaler.fit_transform(pts_3d_rect)

        return pts_3d_rect

    def _2d_to_3d(self, pts):
        points = cv.convertPointsToHomogeneous(pts)
        points = points.reshape(-3, 2)
        rows, cols = points.shape
        c, r = np.meshgrid(np.arange(cols), np.arange(rows))
        points = np.stack([c, r, points])
        points = points.reshape((3, -1))
        points = points.T

        return points
