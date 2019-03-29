import torch
from torch.nn.functional import interpolate
import numpy as np

def upsample_depth(depth, crop_size):
    channels = 1
    height, width = depth.shape
    resized = interpolate(depth.view(1, channels, height, width),
                size=crop_size, mode='bilinear', align_corners=False)
    return resized.view(*crop_size)

def generate_candidates(image_size, depths):
    """Resizes a predicted depth map to the size of its original crop."""

    # Need size in height, width order
    image_size = (image_size[1], image_size[0])

    depths = -torch.sqrt(225 - 20 * depths) + 15
    depths_iter = iter(depths)

    candidates = []

    for idx in range(18):
        crop_ratio = 1 - 0.05 * np.ceil(idx // 2)
        crop_size = tuple(np.round(np.array(image_size) * crop_ratio).astype(np.int))

        next_depth = lambda: upsample_depth(next(depths_iter), crop_size)

        if crop_ratio == 1:
            candidate = next_depth()
        else:
            candidate = torch.zeros(image_size)
            weights = torch.zeros_like(candidate)

            # Upper left
            candidate[:crop_size[0], :crop_size[1]] += next_depth()
            weights[:crop_size[0], :crop_size[1]] += 1

            # Upper right
            candidate[:crop_size[0], image_size[1] - crop_size[1]:] += next_depth()
            weights[:crop_size[0], image_size[1] - crop_size[1]:] += 1

            # Lower left
            candidate[image_size[0] - crop_size[0]:, :crop_size[1]] += next_depth()
            weights[image_size[0] - crop_size[0]:, :crop_size[1]] += 1

            # Lower right
            candidate[image_size[0] - crop_size[0]:, image_size[1] - crop_size[1]:] += next_depth()
            weights[image_size[0] - crop_size[0]:, image_size[1] - crop_size[1]:] += 1

            candidate = candidate / weights / crop_ratio

        if idx % 2 == 1:
            candidate = torch.flip(candidate, (1,))

        candidate = upsample_depth(candidate, (72, 96))

        candidates.append(candidate)

    return candidates
