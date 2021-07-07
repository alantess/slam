from scipy.spatial.transform import Rotation as R
from torchvision import transforms
import torch.nn.functional as F
import numpy as np
import torch
import cv2 as cv

pixel_coords = None

loss = torch.nn.MSELoss()


# Absolute Trajectory Error (Localization Error)
def compute_ate(pred_pose, gt_pose):
    # Reference https://arxiv.org/pdf/2105.14994v1.pdf
    # Loss between the rotation matrix
    assert len(pred_pose) == len(gt_pose)
    return torch.sqrt(
        torch.mean(torch.sum((pred_pose[:, :3] - gt_pose[:, :3])**2, axis=1)))


# Translation Distance
def compute_translation(pred_pose, gt_pose):
    assert len(pred_pose) == len(gt_pose)
    return loss(gt_pose[:, :, -1:], pred_pose[:, :, -1:])


# Projection Loss
def compute_projection(pred, gt, intrinsic):
    # P = kRt
    pred = intrinsic @ pred
    gt = intrinsic @ gt
    return loss(pred, gt)


@torch.no_grad()
def get_depth(
    img,
    model,
    denormalizer,
    device=None,
    disparity=False,
    grayscale=True,
):

    img = img.to(dtype=torch.float32)
    rgb_to_gray = transforms.Grayscale()

    if device:
        model.to(device)
        img = img.to(device)

    output = model(img)
    output = denormalizer(output)
    output = output[0].permute(1, 2, 0)
    if disparity:
        output = 1 / output
    output = output.detach().cpu().numpy().astype(np.uint8)
    output = cv.cvtColor(output, cv.COLOR_RGB2GRAY)
    output = cv.applyColorMap(output, cv.COLORMAP_BONE)
    output = torch.from_numpy(output).permute(2, 0, 1).to(dtype=torch.float32)
    if grayscale:
        output = rgb_to_gray(output)

    return output


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
