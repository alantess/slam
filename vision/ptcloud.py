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

    def init(self):

        # Sets up 3D GUI
        if self.bg_color is None:
            self.bg_color = np.array([0, 0, 0])
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(width=self.width, height=self.height)
        opt = self.vis.get_render_option()
        opt.background_color = self.bg_color

        self.geometry = o3d.geometry.PointCloud()
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

    def run(self, xyz):
        self.add_points(xyz)
        self.vis.run()

    def _get_image(self, d_img):
        return o3d.geometry.Image(d_img[0].squeeze(0).numpy())

    def add_points(self, d_img):
        img = self._get_image(d_img)
        pcd = o3d.geometry.PointCloud.create_from_depth_image(
            depth=img, intrinsic=self.ph_cam)

        self.geometry.points = pcd.points
        self.vis.add_geometry(self.geometry)
        ctr = self.vis.get_view_control()
        ctr.rotate(980.0, 300.0)
        ctr.set_zoom(1.0)
        self.vis.poll_events()
        self.vis.update_renderer()
