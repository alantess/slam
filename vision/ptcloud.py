import open3d as o3d
import numpy as np


class PointCloud(object):
    def __init__(self,
                 k=None,
                 height=700,
                 width=700,
                 model=None,
                 background_color=None,
                 device=None):
        self.k = k
        self.ph_cam = None
        self.model = model
        self.device = device
        self.vis = None
        self.time_step = 0
        self.geometry = None
        self.height = height
        self.width = width
        self.bg_color = background_color
        self.prev = None

    def init(self):

        # Sets up 3D GUI
        if self.bg_color is None:
            self.bg_color = np.array([0, 0, 0])
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(width=self.width, height=self.height)
        opt = self.vis.get_render_option()
        opt.background_color = self.bg_color

        self.geometry = o3d.geometry.PointCloud()

        # self.vis.add_geometry(self.geometry)
        # Set up intrinsics matrix
        fx = self.k[0, 0]
        fy = self.k[1, 1]
        cx = self.k[0, 2]
        cy = self.k[1, 2]

        self.ph_cam = o3d.camera.PinholeCameraIntrinsic(width=self.width,
                                                        height=self.height,
                                                        fx=fx,
                                                        fy=fy,
                                                        cx=cx,
                                                        cy=cy)

    def __del__(self):
        self.vis.destroy_window()

    def run(self, depth):
        self.add_points(depth)
        self.time_step += 1

    def _get_image(self, d_img):
        depth = o3d.geometry.Image(d_img[0].squeeze(0).numpy())
        return depth

    def add_points(self, d_img):
        img = self._get_image(d_img)
        self.geometry.points = self.geometry.create_from_depth_image(
            depth=img, intrinsic=self.ph_cam).points

        self.geometry.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0],
                                 [0, 0, 0, 1]])
        self.vis.add_geometry(self.geometry)
        if self.prev is not None:
            self.vis.add_geometry(self.prev)
        if self.time_step >= 1:
            ctr = self.vis.get_view_control()
            ctr.set_lookat([0, 0, 0])
            ctr.rotate(0.0, 300.0)
            ctr.set_zoom(1.0)
        self.vis.poll_events()
        self.vis.update_renderer()
        self.prev = self.geometry
