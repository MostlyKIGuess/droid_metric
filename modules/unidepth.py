# standard library
from pathlib import Path
from typing import *
import sys, os
# third party
import numpy as np
import torch
from PIL import Image
from unidepth.models import UniDepthV2
# here one can use UnidepthV1 or V2Old as well

__ALL__ = ['UniDepth']

class UniDepth:
    model_: torch.nn.Module
    device: str

    def __init__(
        self,
        model_name: str = 'v2-vitl14',
        device: str = 'cuda'
    ) -> None:
        self.device = device
        # model_name like 'v2-vitl14' should map to 'lpiccinelli/unidepth-v2-vitl14'
        # check this for more info: https://github.com/lpiccinelli-eth/UniDepth/tree/main?tab=readme-ov-file#model-zoo
        model_id = f"lpiccinelli/unidepth-{model_name}"
        
        try:
            self.model_ = UniDepthV2.from_pretrained(model_id)
        except Exception as e:
            print(f"Failed to load model {model_id}: {e}")
            # Try alternative model names or local loading
            raise
        
        self.model_.to(device)
        self.model_.eval()
        
        # Set resolution level for V2 models (optional)
        if hasattr(self.model_, 'resolution_level'):
            self.model_.resolution_level = 9  # Default resolution level

    @torch.no_grad()
    def __call__(
        self,
        rgb_image: Union[np.ndarray, Image.Image, str, Path],
        intrinsic: Optional[Union[str, Path, np.ndarray]] = None,
        d_max: Optional[float] = 300,
        d_min: Optional[float] = 0
    ) -> np.ndarray:
        # read image
        if isinstance(rgb_image, (str, Path)):
            rgb_image = Image.open(rgb_image)
        elif isinstance(rgb_image, np.ndarray):
            rgb_image = Image.fromarray(rgb_image)
        
        # Convert PIL to tensor format expected by UniDepth
        rgb_tensor = torch.from_numpy(np.array(rgb_image)).permute(2, 0, 1).float().to(self.device)
        
        # Handle intrinsic (optional for UniDepth)
        camera_matrix = None
        if intrinsic is not None:
            # get intrinsic
            if isinstance(intrinsic, (str, Path)):
                intrinsic = np.loadtxt(intrinsic)
            
            # Handle your intrinsic format [fx, fy, cx, cy]
            if len(intrinsic) == 4:
                fx, fy, cx, cy = intrinsic
            else:
                # If it's already a 3x3 matrix, extract values
                fx, fy = intrinsic[0, 0], intrinsic[1, 1]
                cx, cy = intrinsic[0, 2], intrinsic[1, 2]
            
            # Create 3x3 camera matrix
            camera_matrix = torch.tensor([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ], dtype=torch.float32).to(self.device)
        
        # Predict depth (camera_matrix can be None for UniDepth)
        predictions = self.model_.infer(rgb_tensor, camera_matrix)
        
        # Extract depth
        pred_depth = predictions["depth"].squeeze().cpu().numpy()
        
        # Apply depth limits
        pred_depth[pred_depth > d_max] = 0
        pred_depth[pred_depth < d_min] = 0
        
        return pred_depth

    @staticmethod
    def gray_to_colormap(depth: np.ndarray) -> np.ndarray:
        """Convert grayscale depth to colormap"""
        import cv2
        depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
        return cv2.cvtColor(depth_colormap, cv2.COLOR_BGR2RGB)
