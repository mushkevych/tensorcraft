import cv2
import numpy as np


def load_image(file_url: str, flag: int = cv2.IMREAD_COLOR) -> np.ndarray:
    """
    :return: image as ndarray of shape:
        * with flag=cv2.IMREAD_COLOR an image is returned in shape [Height x Width x Channels]
          ex: (1024, 1024, 3)
        * with flag=cv2.IMREAD_GRAYSCALE an image is returned in shape [Height x Width]
          ex: (1024, 1024) - with NO COLOR CHANNEL DIMENSION
    """
    image = cv2.imread(file_url, flags=flag)
    return image


def resize_and_pad(image: np.ndarray, target_size: tuple[int, int] = (320, 320)) -> np.ndarray:
    """
    Resizes the image while maintaining aspect ratio and then pads
    it to the target (height, width).

    :param image:
        A NumPy array representing the image. Shape can be [H, W] (grayscale)
        or [H, W, C] (color).
    :param target_size:
        a tuple (height, width), use those exact dimensions.
    :return:
        A resized and padded image with final shape [target_height, target_width]
        (plus channel dimension if color).
    """
    # Handle both int and tuple inputs for target size
    if isinstance(target_size, int):
        target_height, target_width = target_size, target_size
    else:
        target_height, target_width = target_size

    # Get original image dimensions
    original_height, original_width = image.shape[:2]

    # Compute scale such that the image fits within the target_size
    scale_h: float = target_height / original_height
    scale_w: float = target_width / original_width
    scale: float = min(scale_h, scale_w)

    # Determine new dimensions after scaling
    new_height: int = int(original_height * scale)
    new_width: int = int(original_width * scale)

    # Resize the image
    resized_image: np.ndarray = cv2.resize(
        image,
        dsize=(new_width, new_height),
        interpolation=cv2.INTER_AREA
    )

    # Calculate padding for each edge
    delta_width: int = target_width - new_width
    delta_height: int = target_height - new_height

    top: int = delta_height // 2
    bottom: int = delta_height - top
    left: int = delta_width // 2
    right: int = delta_width - left

    # Determine padding value
    padding_value = 0 if resized_image.ndim == 2 else [0, 0, 0]

    # Pad the image
    padded_image: np.ndarray = cv2.copyMakeBorder(
        resized_image,
        top=top, bottom=bottom,
        left=left, right=right,
        borderType=cv2.BORDER_CONSTANT,
        value=padding_value
    )

    return padded_image


def resize_crop_and_pad(image: np.ndarray, target_size: tuple[int, int] = (1024, 1024)) -> np.ndarray:
    """
    Resizes an image while maintaining aspect ratio, crops or pads it to match `target_size`.

    :param image: Input image as a NumPy array (HxWxC for color or HxW for grayscale).
    :param target_size: Final size (height, width) for cropping and/or padding after resizing.
    :return: Image of size `target_size`.
    """
    target_height, target_width = target_size
    original_height, original_width = image.shape[:2]

    if original_height >= target_height and original_width >= target_width:
        # Both dimensions exceed target size; resize while maintaining aspect ratio
        aspect_ratio = original_width / original_height
        if aspect_ratio > 1:
            # Wider than tall, fit height
            new_height = target_height
            new_width = int(new_height * aspect_ratio)
        else:
            # Taller than wide, fit width
            new_width = target_width
            new_height = int(new_width / aspect_ratio)

        resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

        # Crop to target size
        cropped_image = resized_image[:target_height, :target_width]
    else:
        # Handle smaller dimensions: crop and pad
        pad_h = max(0, target_height - original_height)
        pad_w = max(0, target_width - original_width)

        cropped_image = image[:target_height, :target_width]

        # Add padding to target size
        cropped_image = cv2.copyMakeBorder(
            cropped_image,
            top=0, bottom=pad_h,
            left=0, right=pad_w,
            borderType=cv2.BORDER_REPLICATE
        )

    return cropped_image
