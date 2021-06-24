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

        return self.img, tripoints, kpts, matches

    def extract_feats(self, img):
        ret = []
        # Extracts ORB
        feats = cv.goodFeaturesToTrack(
            np.linalg.norm(img, axis=2, ord=2).astype(np.uint8), 4000, 0.02, 3)
        kps = [cv.KeyPoint(x=f[0][0], y=f[0][1], _size=20) for f in feats]

        self.img = cv.drawKeypoints(img,
                                    kps,
                                    None,
                                    color=(248, 186, 255),
                                    flags=0)
        kpts, des = self.orb.compute(img, kps)
        if self.last is not None:
            matches = self.bf.knnMatch(des, self.last['des'], 2)
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
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


class PointMap(object):
    def __init__(self):
        self.array = [0, 0, 0]

    def collect_points(self, tripoints):
        if len(tripoints) > 0:
            array_to_project = np.array([0, 0, 0])

            x_points = [pt for pt in tripoints[0]]
            y_points = [-pt for pt in tripoints[1]]
            z_points = [-pt for pt in tripoints[2]]

            for i in range(tripoints.shape[1]):
                curr_array = np.array([x_points[i], y_points[i], z_points[i]])
                array_to_project = np.vstack((array_to_project, curr_array))

            array_to_project = array_to_project[1:, :]

            return array_to_project
