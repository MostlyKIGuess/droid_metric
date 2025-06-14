import numpy as np
import os
from pathlib import Path
import argparse
from evo.core import transformations as tr

def convert_poses_to_tum(input_dir, output_file):
    """Convert pose matrices to TUM format using EVO's transformation functions"""
    pose_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.txt')])
    
    with open(output_file, 'w') as f_out:
        for i, pose_file in enumerate(pose_files):
            matrix = np.loadtxt(os.path.join(input_dir, pose_file))
            # translation 
            
            tx, ty, tz = matrix[0:3, 3]
            
            # Extract rotation quaternion using EVO's function
            # EVO expects w,x,y,z order for quaternions
            quat = tr.quaternion_from_matrix(matrix)
            qw, qx, qy, qz = quat
            
            #  timestamp tx ty tz qx qy qz qw
            tum_line = f"{i:.6f} {tx:.6f} {ty:.6f} {tz:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}"
            f_out.write(tum_line + "\n")
    
    print(f"Converted {len(pose_files)} poses to TUM format in {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert pose matrices to TUM format")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing pose .txt files")
    parser.add_argument("--output_file", type=str, required=True, help="Output TUM format file")
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_file = Path(args.output_file)
    
    convert_poses_to_tum(input_dir, output_file)
