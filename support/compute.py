from scipy.spatial.transform import Rotation as R
from torchvision import transforms
import numpy as np
import torch
import cv2 as cv

pixel_coords = None
loss_fn = torch.nn.MSELoss()
smooth_loss = torch.nn.SmoothL1Loss()


def compute_pose_loss(pred_pose, gt_pose):
    """
    Inputs:
    Predicted Pose [B,3,4]
    Ground Truth [B,3,4]
    Returns:
    Rotational [B,1] and Translation Error [B,1]
    """
    assert len(pred_pose) == len(gt_pose)
    pred_rot = pred_pose[:, :3, :3]
    pred_translation = pred_pose[:, :, -1:]
    gt_rot = gt_pose[:, :3, :3]
    gt_translation = gt_pose[:, :, -1:]

    rot_err = torch.linalg.norm(gt_rot - pred_rot, axis=1)
    translation_err = torch.linalg.norm(gt_translation - pred_translation,
                                        axis=1)

    return rot_err.mean(), translation_err.mean()


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
