from typing import Literal
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module
from torch.utils.tensorboard import SummaryWriter


class GradCAM:
    """ Class Activation Maps (CAM) or Grad-CAM (Gradient-weighted Class Activation Mapping)
        - mechanism to obtain a *heat map* visualization that highlights areas of an image
          that contribute to a neural network’s decision (whether positive or negative)
    """

    def __init__(self, model: Module, target_layer: Module) -> None:
        """
        Initializes the GradCAM object.

        :param model: The trained model (e.g., EfficientNet).
        :param target_layer: The target layer from which to extract gradients (typically the last convolutional layer).
        """
        self.model: Module = model
        self.target_layer: Module = target_layer
        self.gradients: Optional[Tensor] = None
        self.activations: Optional[Tensor] = None
        self.input_tensor: Optional[Tensor] = None
        self.hook_layers()

    def hook_layers(self) -> None:
        """
        Hooks the forward and backward passes to capture activations and gradients for Grad-CAM.
        """

        # Hook to capture gradients from the target layer
        def backward_hook(module: Module, grad_input: Tuple[Tensor, ...], grad_output: Tuple[Tensor, ...]) -> None:
            self.gradients = grad_output[0]  # Capture the gradient of the output wrt input

        # Hook to capture activations from the target layer
        def forward_hook(module: Module, input: Tuple[Tensor, ...], output: Tensor) -> None:
            self.activations = output  # Capture the activation

        # Register forward and backward hooks
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_heatmap(self, class_idx: int) -> np.ndarray:
        """
        Generates the heatmap based on the activations and gradients for the given class.

        :param class_idx: Index of the target class for which to generate the heatmap.
        :return: Heatmap as a NumPy array.
        """
        # Backpropagate the target class
        if self.input_tensor is None:
            raise ValueError('Input tensor is not set.')

        self.model.zero_grad()
        output: Tensor = self.model(self.input_tensor)

        if output.dim() == 2 or class_idx is None:
            # Check if output is [batch_size] for integer labels
            # No slicing needed; we assume this is the label prediction
            target_output: Tensor = output
        else:
            # Otherwise, slice the output by class index (logits/probabilities case)
            if class_idx is None:
                raise ValueError('Class index must be provided for multi-class output.')
            target_output: Tensor = output[:, class_idx]  # Get the score for the target class

        target_output.backward()

        if self.gradients is None or self.activations is None:
            raise ValueError("Gradients or activations not captured.")

        # Get the gradients and activations
        gradients: Tensor = self.gradients  # Shape: [batch_size, num_filters, h, w]
        activations: Tensor = self.activations  # Shape: [batch_size, num_filters, h, w]

        # Global average pooling over the gradients
        weights: Tensor = torch.mean(gradients, dim=[2, 3])  # Shape: [batch_size, num_filters]

        # Weighted sum of the activations
        heatmap: Tensor = torch.sum(weights[:, :, None, None] * activations, dim=1)  # Shape: [batch_size, h, w]

        # OPTIONAL STEP (for visualization clarity): Apply ReLU to the heatmap
        heatmap = F.relu(heatmap)
        np_heatmap = heatmap.squeeze().cpu().detach().numpy()

        # Normalize the heatmap between 0 and 1 for visualization
        np_heatmap = (np_heatmap - np.min(np_heatmap)) / max(np.max(np_heatmap) - np.min(np_heatmap), 1e-8)
        return np_heatmap

    def __call__(self, input_tensor: Tensor, class_idx: int) -> np.ndarray:
        """
        Callable method to generate Grad-CAM for the input image and target class.

        :param input_tensor: Input image tensor of shape [1, 1, H, W] for grayscale.
        :param class_idx: Target class index (e.g., positive/negative).
        :return: The generated heatmap as a NumPy array.
        """
        self.input_tensor = input_tensor  # Store input for the backward pass
        return self.generate_heatmap(class_idx)


def instantiate_gram_cam(model: Module) -> GradCAM:
    # Assume `model` is the EfficientNet-B0 model
    # Get target layer (last conv layer of EfficientNet-B0, e.g., `_blocks[-1]`)
    target_layer = model._blocks[-1]
    grad_cam = GradCAM(model=model, target_layer=target_layer)
    return grad_cam


def write_cam_to_tensorboard(
    writer: SummaryWriter,
    grad_cam: GradCAM,
    img_tensor: Tensor,
    target_class: Literal[0, 1],
    img_idx: int,
) -> None:
    """
    :param writer: Tensorboard SummaryWriter object.
    :param grad_cam: initialized GradCAM object.
    :param img_tensor: Image greyscale as a NumPy array of shape [1, H, W]
    :param target_class: Target class index (e.g., positive=1 / negative=0).
    :param img_idx: id of the image in the Tensorboard.
    """
    input_tensor = img_tensor.unsqueeze(0)  # Add batch dimension (shape: [1, 1, H, W])

    # Generate the heatmap
    heatmap = grad_cam(input_tensor, target_class)

    # Convert heatmap to RGB format for TensorBoard logging
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap_colored = np.transpose(heatmap_colored, (2, 0, 1))  # Convert to [3, H, W] format for TensorBoard

    # Optionally, overlay the heatmap on the original greyscale image
    original_img: np.ndarray = img_tensor.squeeze().cpu().numpy()  # Convert to 2D NumPy array (H, W)
    if original_img.max() - original_img.min() > 0:
        # Normalize to [0, 1]
        original_img = (original_img - original_img.min()) / (original_img.max() - original_img.min())
    else:
        # Set to a constant value for uniform images (i.e. a blank image)
        np.zeros_like(original_img)
    original_img = np.uint8(255 * original_img)  # Scale to [0, 255]

    # Convert greyscale image to 3-channel RGB image
    original_img_rgb = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)  # Shape will now be [240, 240, 3]

    # Resize the heatmap to match the original_img_rgb shape - [240, 240, 3]
    heatmap_resized = cv2.resize(heatmap_colored.transpose(1, 2, 0), (original_img.shape[1], original_img.shape[0]))

    # Overlay the resized heatmap on the original RGB image
    overlayed_img = cv2.addWeighted(src1=original_img_rgb, alpha=0.6, src2=heatmap_resized, beta=0.4, gamma=0)
    final_heatmap = overlayed_img.transpose(2, 0, 1)  # Shape: [3, H, W]

    writer.add_image(f'Grad-CAM-{target_class}', img_tensor=final_heatmap, global_step=img_idx, dataformats='CHW')
