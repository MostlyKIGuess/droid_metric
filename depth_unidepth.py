# standard library
from pathlib import Path
from typing import *
import os
# third party
import argparse
from tqdm import tqdm
import numpy as np
import cv2
# UniDepth
from modules import UniDepth

def main(
    input_images: Union[str, Path],
    output_dir: Union[str, Path],
    intrinsic: Optional[Union[str, Path]] = None,
    d_max: Optional[float] = 300.0,
    overwrite: Optional[bool] = True,
    save_colormap: Optional[bool] = False,
    model_name: Optional[str] = 'v2-vitl14'
) -> None:
    # load intrinsic (optional for UniDepth)
    intr = None
    if intrinsic is not None:
        intr = np.loadtxt(intrinsic)[:4]
    
    # init UniDepth
    metric = UniDepth(model_name=model_name)
    # load images
    image_dir = Path(input_images).resolve()
    
    # Check if directory exists
    if not image_dir.exists():
        print(f"Error: Directory {image_dir} does not exist!")
        return
    
    # More comprehensive image file pattern matching
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']
    images = []
    for ext in image_extensions:
        images.extend(image_dir.glob(ext))
    images = sorted(images)
    
    print(f"Found {len(images)} images in {image_dir}")
    if len(images) == 0:
        print("Available files in directory:")
        for file in image_dir.iterdir():
            if file.is_file():
                print(f"  {file.name}")
        return
    
    # create output dir
    out_dir = Path(output_dir).resolve()
    os.makedirs(out_dir, exist_ok=True)
    # create colormap dir
    color_dir = out_dir / 'colormap'
    if save_colormap:
        os.makedirs(color_dir, exist_ok=True)

    print(f'Processing {len(images)} images...')
    for image in tqdm(images):
        if overwrite or not (out_dir / f'{image.stem}.npy').exists():
            depth = metric(rgb_image=image, intrinsic=intr, d_max=d_max)
            # save original depth
            np.save(str(out_dir / f'{image.stem}.npy'), depth)
            # save colormap
            if save_colormap:
                depth_color = metric.gray_to_colormap(depth)
                depth_color = cv2.cvtColor(depth_color, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(color_dir / f'{image.stem}.png'), depth_color)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Running UniDepth')
    parser.add_argument("--images", help='dir for rgb image files', type=str, required=True)
    parser.add_argument("--intr", help='intrinsic txt file, contained [fx, fy, cx, cy] (optional for UniDepth)', type=str, required=False, default=None)
    parser.add_argument("--out", help='dir for output depth', type=str, default='')
    parser.add_argument("--out-colormap", action='store_true', help='save colormap for depth', default=False)
    parser.add_argument("--dmax", help='max depth', type=float, default=300.0)
    parser.add_argument('--skip-existed', action='store_true', help='skip existing depth file', default=False)
    parser.add_argument("--model-name", type=str, default='v2-vitl14', 
                        choices=['v2-vitl14', 'v2-cnvnxt-large', 'v2-vits14'], 
                        help='UniDepth model variant')
    args = parser.parse_args()

    main(
        input_images=args.images,
        output_dir=args.out,
        intrinsic=args.intr,
        d_max=args.dmax,
        save_colormap=args.out_colormap,
        model_name=args.model_name,
        overwrite=not args.skip_existed
    )
