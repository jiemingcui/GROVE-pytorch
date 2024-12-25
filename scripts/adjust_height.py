# Aim:     Adjust root height of poses to ensure that the frame with the lowest foot height touches the ground.
# Method:  Create two empty objects at the feet, and after forward kinematics calculations in Blender, obtain the foot positions.
# Modules: bpy, sRot
# Args:    folder_path(str), output_file(str)

import argparse
import bpy
import joblib
import numpy as np
from tqdm import tqdm
import os
from scipy.spatial.transform import Rotation as sRot
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PoseHeightAdjuster:
    def __init__(self, folder_path='data/raw_data/smplx322/idea400', output_file='data/pose_data/pose_data_idea400.joblib'):
        self.folder_path = folder_path
        self.output_file = output_file
        self.pose_data = None
        self.SMPL_BONE_ORDER_NAMES = [
            "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee", 
            "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot", 
            "neck", "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder", 
            "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_pinky1", "right_pinky1"
        ]
        self.arrays_list = None

    def load_pose_data(self):
        """Load pose data from the specified folder and convert it to Euler angles."""
        arrays_list = []
        logging.info("Loading pose data from folder: %s", self.folder_path)

        for file_name in os.listdir(self.folder_path):
            file_path = os.path.join(self.folder_path, file_name)
            array = np.load(file_path)
            pose_aa = array[:, 0:72]
            rot_matrices = sRot.from_rotvec(pose_aa.reshape(-1, 3))
            pose_euler = rot_matrices.as_euler('xyz').reshape(-1, 72)
            pose_data_each_sequence = np.concatenate((pose_euler, array[:, 311].reshape(-1, 1)), axis=1)
            arrays_list.append(pose_data_each_sequence)

        self.arrays_list = arrays_list
        self.pose_data = np.concatenate(arrays_list, axis=0)
        logging.info("Pose data loaded successfully.")

    def adjust_root_height(self):
        """Adjust the root height of poses to ensure the lowest foot height is on the ground."""
        filepath = "assets/smplx_female.blend"
        bpy.ops.wm.open_mainfile(filepath=filepath)
        frame_idx = 0
        counter = 0
        logging.info("Starting root height adjustment.")

        for pose_data_each_sequence in tqdm(self.arrays_list, desc="Processing"):
            if counter == 100:
                bpy.ops.wm.open_mainfile(filepath=filepath)
                counter = 0
            counter += 1
            min_height = 100
            
            bones = bpy.data.objects['SMPLX-female'].pose.bones        
            for pose_idx in range(len(pose_data_each_sequence)):    
                for i, joint_name in enumerate(self.SMPL_BONE_ORDER_NAMES):
                    bones[joint_name].rotation_euler = pose_data_each_sequence[pose_idx][3*i:3*i+3]

                left_hand = ["left_index1", "left_middle1", "left_ring1", "left_pinky1"]
                right_hand = ["right_index1", "right_middle1", "right_ring1", "right_pinky1"]
                for fingers in left_hand:
                    bones[fingers].rotation_euler = pose_data_each_sequence[pose_idx][66:69]
                for fingers in right_hand:
                    bones[fingers].rotation_euler = pose_data_each_sequence[pose_idx][69:72]

                bones["root"].location[2] = pose_data_each_sequence[pose_idx][72]
                bpy.context.view_layer.depsgraph.update()
                bones = bpy.data.objects['SMPLX-female'].pose.bones

                left_foot = bones["left_foot"]
                right_foot = bones["right_foot"]

                # Convert from pose to world coordinates
                matrix_world_left = bpy.data.objects['SMPLX-female'].convert_space(
                    pose_bone=left_foot, matrix=left_foot.matrix, from_space='POSE', to_space='WORLD'
                )
                empty_l = bpy.data.objects.new(left_foot.name, None)
                empty_l.name = left_foot.name
                empty_l.matrix_world = matrix_world_left
                left_foot_location = empty_l.location

                matrix_world_right = bpy.data.objects['SMPLX-female'].convert_space(
                    pose_bone=right_foot, matrix=right_foot.matrix, from_space='POSE', to_space='WORLD'
                )
                empty_r = bpy.data.objects.new(right_foot.name, None)
                empty_r.name = right_foot.name
                empty_r.matrix_world = matrix_world_right
                right_foot_location = empty_r.location
                
                bpy.data.objects.remove(empty_l, do_unlink=True)
                bpy.data.objects.remove(empty_r, do_unlink=True)

                if left_foot_location[2] < min_height:
                    min_height = left_foot_location[2]
                if right_foot_location[2] < min_height:
                    min_height = right_foot_location[2]
            
            logging.info("Minimum height for current batch: %f", min_height)
            pose_data_each_sequence[:, 72] += 1.644 - min_height
            self.pose_data[frame_idx:frame_idx + len(pose_data_each_sequence), 72] = pose_data_each_sequence[:, 72]
            frame_idx += len(pose_data_each_sequence)
            
        logging.info("Root height adjustment completed.")

    def save_pose_data(self):
        """Save the adjusted pose data to a file."""
        joblib.dump(self.pose_data, self.output_file)
        logging.info("Adjusted pose data saved to: %s", self.output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Adjust root height of poses in Blender.')
    parser.add_argument('--folder_path', type=str, default='data/raw_data/smplx322/idea400', help='Directory containing pose data files')
    parser.add_argument('--output_file', type=str, default='data/pose_data/pose_data_idea400.joblib', help='Output file for adjusted pose data')

    args = parser.parse_args()

    adjuster = PoseHeightAdjuster(folder_path=args.folder_path, output_file=args.output_file)
    adjuster.load_pose_data()
    adjuster.adjust_root_height()
    adjuster.save_pose_data()
