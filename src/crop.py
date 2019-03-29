from PIL import Image
import numpy as np
import torchvision.transforms.functional as TF

def crop_projection_mask(image):
    """Crops an image from the NYU dataset to the central region where the
    Kinect depth signal is most accurate."""

    return image.crop((40, 44, 601, 471))

def generate_crops(original_image):
    """Generates a number of crops from a given image, which can be used to
    estimate depth at different levels of detail.
    """

    crops = []

    input_size = np.array(original_image.size)

    for idx in range(18):
        image = original_image

        if idx % 2 == 0:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        crop_ratio = 1 - 0.05 * np.ceil(idx // 2)

        if crop_ratio == 1:
            crops.append(image)
        else:
            crop_size = np.round(input_size * crop_ratio)

            upper_left = image.crop((
                0, 0,
                crop_size[0], crop_size[1]
            ))
            crops.append(upper_left)

            upper_right = image.crop((
                input_size[0] - crop_size[0],
                0, input_size[0], crop_size[1]
            ))
            crops.append(upper_right)

            lower_left = image.crop((
                0, input_size[1] - crop_size[1],
                crop_size[0], input_size[1]
            ))
            crops.append(lower_left)

            lower_right = image.crop((
                input_size[0] - crop_size[0], input_size[1] - crop_size[1],
                input_size[0], input_size[1]
            ))
            crops.append(lower_right)

    return list(map(process_image, crops))

def process_image(image):
    """Converts a color image to a tensor which can be fed into
    the depth estimation network.
    """

    # Resize to the standard ImageNet input dimension
    image = image.resize((224, 224), resample=Image.BILINEAR)

    # Convert to BGR Format
    r, g, b = image.split()
    image = Image.merge('RGB', (b, g, r))

    tensor = TF.to_tensor(image)

    # Normalize the input
    tensor = TF.normalize(tensor,
        mean=(104 / 255, 117 / 255, 123 / 255),
        std=(1, 1, 1)
    )

    return tensor
