# Aim:     Render poses in Blender using motion data from different angles.
# Method:  Load motion data and a Blender scene, set the character's pose based on the motion data, and render images 
#          from multiple viewpoints. The rendering process is managed by a dedicated class that handles pose adjustments 
#          and rendering settings.
# Modules: numpy, os, bpy, time, math, joblib, scipy.spatial.transform (Rotation)
# Args:    motion_data (str): Path to the motion data file in joblib format.
#          blend_file (str): Path to the Blender file containing the character model.
#          output_dir (str): Directory where rendered images will be saved.


import numpy as np
import os
import bpy
import time
import math
import joblib
import logging
from tqdm import tqdm
from scipy.spatial.transform import Rotation as sRot
import argparse
import sys
from contextlib import contextmanager

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

SMPL_BONE_ORDER_NAMES = [
        "pelvis", "left_hip", "right_hip", "spine1", 
        "left_knee", "right_knee", "spine2", "left_ankle", 
        "right_ankle", "spine3", "left_foot", "right_foot", 
        "neck", "left_collar", "right_collar", "head", 
        "left_shoulder", "right_shoulder", "left_elbow", 
        "right_elbow", "left_wrist", "right_wrist", 
        "left_pinky1", "right_pinky1"
        ]


@contextmanager
def stdout_redirected(to=os.devnull):
    """Redirect stdout to a specified file."""
    fd = sys.stdout.fileno()
    with os.fdopen(os.dup(fd), 'w') as old_stdout:
        with open(to, 'w') as file:
            os.dup2(file.fileno(), fd)
            sys.stdout = os.fdopen(fd, 'w')
            try:
                yield
            finally:
                os.dup2(old_stdout.fileno(), fd)

class PoseRenderer:
    """Class to handle rendering poses in Blender."""
    
    def __init__(self, motion_data_path, blend_file_path, output_directory):
        """Initialize the PoseRenderer with motion data and Blender scene."""
        self.motion_data = joblib.load(motion_data_path)
        self.output_directory = output_directory
        self.bones = self.load_blender_scene(blend_file_path)
        logging.info("Initialized PoseRenderer.")

    def load_blender_scene(self, blend_file_path):
        """Load the Blender scene from the specified file."""
        bpy.ops.wm.open_mainfile(filepath=blend_file_path)
        bpy.context.scene.render.engine = 'BLENDER_EEVEE'
        scene = bpy.context.scene
        scene.render.threads = 2
        scene.eevee.use_shadows = True
        scene.eevee.use_gtao = True
        scene.render.image_settings.file_format = 'PNG'
        scene.render.image_settings.color_mode = 'RGB'
        scene.render.resolution_x = 224
        scene.render.resolution_y = 224
        logging.info("Blender scene loaded.")
        return bpy.data.objects['SMPLX-female'].pose.bones

    def render_pose_range(self, pose_range):
        """Render a range of poses."""
        for pose_idx in tqdm(range(pose_range[0], pose_range[1])):
            self.set_pose(pose_idx)
            self.render_views(pose_idx)

    def set_pose(self, pose_idx):
        """Set the pose for the specified index."""
        for i, joint_name in enumerate(SMPL_BONE_ORDER_NAMES):
            self.bones[joint_name].rotation_euler = self.motion_data[pose_idx][3*i:3*i+3]
        # Set hand bones
        self.set_hand_bones(pose_idx)

    def set_hand_bones(self, pose_idx):
        """Set the hand bones for the specified pose index."""
        left_hand = ["left_index1", "left_middle1", "left_ring1", "left_pinky1"]
        right_hand = ["right_index1", "right_middle1", "right_ring1", "right_pinky1"]
        for fingers in left_hand:
            self.bones[fingers].rotation_euler = self.motion_data[pose_idx][66:69]
        for fingers in right_hand:
            self.bones[fingers].rotation_euler = self.motion_data[pose_idx][69:72]

    def render_views(self, pose_idx):
        """Render the pose from different views."""
        views = {
            "front": math.pi,
            "side": math.pi * 3 / 4,
            "oblique": math.pi / 2,
            "rear_side": math.pi / 4,
            "rear": 0
        }
        with stdout_redirected():
            for view_name, rotation in views.items():
                self.bones['pelvis'].rotation_euler[2] = rotation
                self.bones["root"].location[2] = self.motion_data[pose_idx][72]
                bpy.context.scene.render.filepath = os.path.join(self.output_directory, f"{view_name}/{pose_idx}.png")
                bpy.ops.render.render(write_still=True, use_viewport=True)

def main(args):
    """Main function to parse arguments and start rendering."""
    renderer = PoseRenderer(args.motion_data, args.blend_file, args.output_dir)
    pose_length = len(renderer.motion_data)
    
    # Render all poses sequentially
    renderer.render_pose_range((0, pose_length))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Render poses in Blender.")
    parser.add_argument('--motion_data', type=str, default='data/pose_data/pose_data_idea400.joblib', help='Path to the motion data file.')
    parser.add_argument('--blend_file', type=str, default='assets/smplx_female.blend', help='Path to the Blender file.')
    parser.add_argument('--output_dir', type=str, default='data/rendered_images/idea400/', help='Output directory for rendered images.')
    args = parser.parse_args()
    main(args)
