import numpy as np
import cv2 as cv
from sklearn.preprocessing import MinMaxScaler


class FeatureExtractor(object):
    def __init__(self, H, W):
        self.H = H
        self.W = W
        self.scaler = MinMaxScaler()
        self.orb = cv.ORB_create()
        # Camera intrinsics and extrinsics
        self.P = self._calibrate()

        # Camera intrinsics and extrinsics
        self.c_u = self.P[0, 2]
        self.c_v = self.P[1, 2]
        self.f_u = self.P[0, 0]
        self.f_v = self.P[1, 1]
        self.b_x = self.P[0, 3] / (-self.f_u)  # relative
        self.b_y = self.P[1, 3] / (-self.f_v)

    def _calibrate(self):
        res = np.eye(4)
        res[0, 2] = self.W // 2
        res[1, 2] = self.H // 2
        return res

    # Extracts features
    def extract(self, img):
        # img = self.get_direct_matches(img)

        # img, kps_feat = self.get_orb_feats(img)

        # pts_orb = cv.KeyPoint_convert(kps_feat)

        # xyz = self.lidar_generation(pts_orb)

        img, xyz = self.get_direct_lines(img)

        pts = self.generated_3d_lines(xyz)

        return img, pts

    def get_orb_feats(self, img):
        # Extracts ORB
        feats = cv.goodFeaturesToTrack(
            np.mean(img, axis=2).astype(np.uint8), 3000, 0.01, 4)
        kps = [cv.KeyPoint(x=f[0][0], y=f[0][1], _size=20) for f in feats]
        orbs, des = self.orb.compute(img, kps)
        img = cv.drawKeypoints(img, kps, None, color=(248, 186, 255), flags=0)

        return img, orbs

    def get_direct_matches(self, img):
        # Splits images in Half
        left = img[:, :int(self.W / 2)].astype(np.uint8)
        right = img[:, int(self.W / 2):].astype(np.uint8)
        # Brute-Force Matching with ORB Descriptors
        bf = cv.BFMatcher(cv.NORM_HAMMING)

        kp1, des1 = self.orb.detectAndCompute(left, None)
        kp2, des2 = self.orb.detectAndCompute(right, None)

        matches = bf.knnMatch(des1, des2, 2)
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append([m])

        img = cv.drawMatchesKnn(
            left,
            kp1,
            right,
            kp2,
            good,
            None,
            flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        return img

    def get_direct_lines(self, img):
        denoise_lines = []
        lsd = cv.ximgproc.createFastLineDetector()
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        lines = lsd.detect(img)
        avg = np.mean(lines)
        denoise_lines = np.array(denoise_lines)
        img = lsd.drawSegments(img, denoise_lines)
        return img, denoise_lines

    def generated_3d_lines(self, lines):
        # reshape to UxV
        lines = lines.reshape(2, -1)
        lines = lines.T
        points = np.ones((lines.shape[0], 3))

        points[:, 0] = lines[:, 0]
        points[:, 1] = lines[:, 1]

        points = self.lidar_generation(points)
        return points

    def lidar_generation(self, pts):
        uv_depth = self._2d_to_3d(pts)
        # uv_depth = self.scaler.fit_transform(uv_depth)
        n = uv_depth.shape[0]
        # x = (uv_depth[:, 0] - self.c_u) / self.f_u
        # y = (uv_depth[:, 1] - self.c_v) / self.f_v

        x = ((uv_depth[:, 0]) * self.c_u / self.f_u) + (
            (uv_depth[:, 0]) * uv_depth[:, 2] / self.f_u)
        y = ((uv_depth[:, 1]) * self.c_v / self.f_v) + (
            (uv_depth[:, 1]) * uv_depth[:, 2] / self.f_v)

        pts_3d_rect = np.zeros((n, 3))
        pts_3d_rect[:, 0] = x
        pts_3d_rect[:, 1] = y
        pts_3d_rect[:, 2] = uv_depth[:, 2]

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
