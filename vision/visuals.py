import numpy as np
import torch
from torchvision import transforms
from common.helpers.support import apply_sharpen_filter
import cv2 as cv
import open3d as o3d
from slam.extract.features import FeatureExtractor

W, H = 720, 480
VIDEO = '../etc/videos/driving.mp4'


class FeatDisplay(object):
    def __init__(self, model=None, device=None):
        # Gui Settings for Point Cloud
        self.model = model
        self.device = device
        if self.model:
            self.model.load()
            self.model.to(device)
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(width=W, height=H, top=600, left=650)
        opt = self.vis.get_render_option()
        opt.background_color = np.asarray([0, 0, 0])
        ctr = self.vis.get_view_control()
        ctr.set_lookat([0, 1, 0])
        ctr.set_front([1, 0, 0])
        ctr.set_up([0, 0, 1])
        ctr.set_zoom(0.25)
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(1.5)

        self.geometry = o3d.geometry.PointCloud()
        self.vis.add_geometry(frame)
        self.vis.clear_geometries()
        self.vis.add_geometry(self.geometry)
        self.extractor = FeatureExtractor(H, W)
        # Open Video
        self.cap = cv.VideoCapture(VIDEO)

    def show_with_model(self):
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((512, 512)),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        img_tensor = torch.zeros((1, 3, 512, 512),
                                 device=self.device,
                                 dtype=torch.float32)
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            img_tensor[0] = preprocess(frame)
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    mask = self.model(img_tensor)
                    mask = mask[0].permute(1, 2, 0)
                    mask = mask.mul(255).clamp(0, 255)
                mask = mask.detach().cpu().numpy().astype(np.float32)
                frame = apply_sharpen_filter(mask, alpha=50).astype(np.uint8)

            frame = cv.resize(frame, (W, H))
            # Display Output
            frame = cv.cvtColor(frame, cv.COLOR_GRAY2RGB)
            # img = cv.applyColorMap(frame, cv.COLORMAP_TWILIGHT_SHIFTED)
            img, xyz = self.extractor.extract(frame)
            self.display_lidar(xyz)
            cv.imshow('frame', img)
            if cv.waitKey(1) == ord('q'):
                break

        self.vis.run()
        self.vis.destroy_window()
        cv.destroyAllWindows()

    def show(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            frame = cv.resize(frame, (W, H))
            img, xyz = self.extractor.extract(frame)
            self.display_lidar(xyz)
            cv.imshow('frame', img)
            if cv.waitKey(1) == ord('q'):
                break

        self.vis.run()
        self.vis.destroy_window()
        cv.destroyAllWindows()

    def display_lidar(self, xyz):
        self.geometry.points = o3d.utility.Vector3dVector(xyz)
        self.vis.update_geometry(self.geometry)
        self.vis.update_renderer()
        self.vis.poll_events()
