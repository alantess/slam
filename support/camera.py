import torch
from torch import Tensor


class CameraProjector(object):
    """
    Takes in a depth map and returns a Nx3 Point Cloud
    Args:
        (K) intrinsics_mat: 3x4 Matrix
            fx 0 cx
            0  fy cy
            0  0  1
    """
    def __init__(self, intrinsics_mat):
        self.K = intrinsics_mat
        self.pixel_coords = None
        self.depth_map = None

    def set_id_grid(self, depth: Tensor) -> None:
        b, h, w = depth.size()
        # Traverse though the X and Y of the pixel
        i_range = torch.arange(0, h).view(1, h,
                                          1).expand(1, h,
                                                    w).type_as(depth)  # 1xHxW
        j_range = torch.arange(0, w).view(1, 1,
                                          w).expand(1, h,
                                                    w).type_as(depth)  # 1xHxW
        ones = torch.ones(1, h, w).type_as(depth)
        self.pixel_coords = torch.stack((j_range, i_range, ones), dim=1)

    def pixel_to_cam(self, depth_map: Tensor):
        """
        Converts a depth map into pixel coordinates  
        s[u,v,1] = k[Xc,Yc,Zc] --> [u,v,1] = 1/s[Xc,Yc,Zc] 
        Args:
            depth_map [B,H,W]
        Returns:
            (Nx3) Perspective Projection  
        """
        depth_map = 1 / depth_map
        inv = self.K.inverse()
        b, h, w = depth_map.size()
        if (self.pixel_coords is None) or self.pixel_coords.size(2) < h:
            self.set_id_grid(depth_map)

        current_pixel_coords = self.pixel_coords[:, :, :h, :w].expand(
            b, 3, h, w).reshape(b, 3, -1)  # [B,3,HxW]

        cam_coords = (inv @ current_pixel_coords).reshape(b, 3, h, w)
        uv = cam_coords * depth_map.unsqueeze(1)
        uv = uv.flatten(2)[0].permute(1, 0)
        return uv.numpy()
