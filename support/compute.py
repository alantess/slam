from scipy.spatial.transform import Rotation as R
import torch.nn.functional as F
import torch


def set_id_grid(depth):
    global pixel_coords
    b, h, w = depth.size()
    i_range = torch.arange(0, h).view(1, h,
                                      1).expand(1, h,
                                                w).type_as(depth)  # [1, H, W]
    j_range = torch.arange(0, w).view(1, 1,
                                      w).expand(1, h,
                                                w).type_as(depth)  # [1, H, W]
    ones = torch.ones(1, h, w).type_as(depth)

    pixel_coords = torch.stack((j_range, i_range, ones), dim=1)  # [1, 3, H, W]


def cam2pixel(cam_coords, proj_c2p_rot, proj_c2p_tr, padding_mode):
    """Transform coordinates in the camera frame to the pixel frame.
    Args:
        cam_coords: pixel coordinates defined in the first camera coordinates system -- [B, 4, H, W]
        proj_c2p_rot: rotation matrix of cameras -- [B, 3, 4]
        proj_c2p_tr: translation vectors of cameras -- [B, 3, 1]
    Returns:
        array of [-1,1] coordinates -- [B, 2, H, W]
    """
    b, _, h, w = cam_coords.size()
    cam_coords_flat = cam_coords.reshape(b, 3, -1)  # [B, 3, H*W]
    if proj_c2p_rot is not None:
        pcoords = proj_c2p_rot @ cam_coords_flat
    else:
        pcoords = cam_coords_flat

    if proj_c2p_tr is not None:
        pcoords = pcoords + proj_c2p_tr  # [B, 3, H*W]
    X = pcoords[:, 0]
    Y = pcoords[:, 1]
    Z = pcoords[:, 2].clamp(min=1e-3)

    # Normalized, -1 if on extreme left, 1 if on extreme right (x = w-1) [B, H*W]
    X_norm = 2 * (X / Z) / (w - 1) - 1
    Y_norm = 2 * (Y / Z) / (h - 1) - 1  # Idem [B, H*W]

    pixel_coords = torch.stack([X_norm, Y_norm], dim=2)  # [B, H*W, 2]
    return pixel_coords.reshape(b, h, w, 2)


def pixel2cam(depth, intrinsics_inv):
    global pixel_coords
    """Transform coordinates in the pixel frame to the camera frame.
    Args:
        depth: depth maps -- [B, H, W]
        intrinsics_inv: intrinsics_inv matrix for each element of batch -- [B, 3, 3]
    Returns:
        array of (u,v,1) cam coordinates -- [B, 3, H, W]
    """
    b, h, w = depth.size()
    if (pixel_coords is None) or pixel_coords.size(2) < h:
        set_id_grid(depth)
    current_pixel_coords = pixel_coords[:, :, :h, :w].expand(
        b, 3, h, w).reshape(b, 3, -1)  # [B, 3, H*W]
    cam_coords = (intrinsics_inv @ current_pixel_coords).reshape(b, 3, h, w)
    return cam_coords * depth.unsqueeze(1)


def pose_to_mat(vec, method='euler'):
    """
    Input [Batch, 6]
    Returns: [Batch,3,4]
    """
    translation = vec[:, :3].unsqueeze(-1)
    rot = vec[:, 3:]

    r = R.from_euler('zyx', rot)
    if method == 'euler':
        rot_mat = R.as_matrix(r)
    rot_mat = torch.from_numpy(rot_mat)

    transformation = torch.cat([rot_mat, translation],
                               dim=2)  # Shapes to [B,3,4]

    return transformation


def inverse_warp(img,
                 depth,
                 pose,
                 intrinsics,
                 rotation_mode='euler',
                 padding_mode='zeros'):
    """
    Inverse warp a source image to the target image plane.
    Args:
        img: the source image (where to sample pixels) -- [B, 3, H, W]
        depth: depth map of the target image -- [B, H, W]
        pose: 6DoF pose parameters from target to source -- [B, 6]
        intrinsics: camera intrinsic matrix -- [B, 3, 3]
    Returns:
        projected_img: Source image warped to the target image plane
        valid_points: Boolean array indicating point validity
    """

    batch_size, _, img_height, img_width = img.size()

    cam_coords = pixel2cam(depth, intrinsics.inverse())  # [B,3,H,W]

    pose_mat = pose_to_mat(pose, rotation_mode)  # [B,3,4]

    # Get projection matrix for tgt camera frame to source pixel frame
    proj_cam_to_src_pixel = intrinsics @ pose_mat  # [B, 3, 4]

    rot, tr = proj_cam_to_src_pixel[:, :, :3], proj_cam_to_src_pixel[:, :, -1:]
    src_pixel_coords = cam2pixel(cam_coords, rot, tr,
                                 padding_mode)  # [B,H,W,2]
    projected_img = F.grid_sample(img,
                                  src_pixel_coords,
                                  padding_mode=padding_mode)

    valid_points = src_pixel_coords.abs().max(dim=-1)[0] <= 1

    return projected_img, valid_points
