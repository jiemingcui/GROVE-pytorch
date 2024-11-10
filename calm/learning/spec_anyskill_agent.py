import copy
from gym import spaces
import numpy as np
import os
import yaml
import wandb
from scipy.spatial.transform import Rotation as R

from rl_games.common import a2c_common

import torch
# from torch.nn.functional import cosine_similarity
import torch.nn.functional as F

import learning.common_agent as common_agent
import learning.calm_agent as calm_agent
import learning.calm_models as calm_models
import learning.calm_network_builder as calm_network_builder
from utils import anyskill
from utils import rendered


def transform_tensor(tensor, tensor_root):
    # tensor: [1024, 28]
    # tensor_root: [1024, 3]

    # Initialize the new tensor with zeros
    new_tensor = torch.zeros([1024, 42], device=tensor.device, dtype=tensor.dtype)

    # Mapping from old indices to new indices
    new_positions = []
    new_idx = 0
    old_idx = 0
    while new_idx < 42 and old_idx < 28:
        if old_idx == 9:
            # Before old_idx 9, insert one zero
            new_positions.append(None)
            new_idx += 1

            # Map old_idx 9
            new_positions.append(old_idx)
            new_idx += 1
            old_idx += 1

            # After old_idx 9, insert four zeros
            for _ in range(4):
                new_positions.append(None)
                new_idx += 1
        elif old_idx == 13:
            # Before old_idx 13, insert one zero
            new_positions.append(None)
            new_idx += 1

            # Map old_idx 13
            new_positions.append(old_idx)
            new_idx += 1
            old_idx += 1

            # After old_idx 13, insert four zeros
            for _ in range(4):
                new_positions.append(None)
                new_idx += 1
        elif old_idx == 17 or old_idx == 24:
            # Before old_idx, insert one zero
            new_positions.append(None)
            new_idx += 1

            # Map old_idx
            new_positions.append(old_idx)
            new_idx += 1
            old_idx += 1

            # After old_idx, insert one zero
            new_positions.append(None)
            new_idx += 1
        else:
            # Map old_idx
            new_positions.append(old_idx)
            new_idx += 1
            old_idx += 1

    # Build the new tensor based on the mapping
    for idx, pos in enumerate(new_positions):
        if pos is not None:
            new_tensor[:, idx] = tensor[:, pos]

    # tensor_root[:, 1] *= -1
    root_rot = R.from_euler('xyz', tensor_root, degrees=False)  # Convert root axis
    tensor_root = torch.tensor(tensor_root, device=new_tensor.device, dtype=new_tensor.dtype)

    inverse_rot = R.from_euler('xyz', torch.tensor([90, 0, 180]), degrees=True)
    res_rot = inverse_rot * root_rot  # Combine
    root_euler = res_rot.as_euler('xyz', degrees=False)
    root_euler = torch.tensor(root_euler, device=new_tensor.device, dtype=new_tensor.dtype)

    # Concatenate tensor_root with the new tensor along dimension 1
    concat_tensor = torch.cat([tensor_root, new_tensor], dim=1)  # Shape: [1024, 45]

    # Reshape to [1024, 15, 3]
    final_tensor = concat_tensor.view(1024, 15, 3)

    return final_tensor


def quat_to_angle(quaternion):
    q0, q1, q2, q3 = quaternion[:, 0], quaternion[:, 1], quaternion[:, 2], quaternion[:, 3]

    # Roll (x-axis rotation)
    sinr_cosp = 2 * (q0 * q1 + q2 * q3)
    cosr_cosp = 1 - 2 * (q1 * q1 + q2 * q2)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (q0 * q2 - q3 * q1)
    pitch = torch.where(torch.abs(sinp) >= 1, torch.sign(sinp) * (torch.pi / 2), torch.asin(sinp))

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (q0 * q3 + q1 * q2)
    cosy_cosp = 1 - 2 * (q2 * q2 + q3 * q3)
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return torch.stack([roll, pitch, yaw], dim=-1)

def reorg(tensor):
    new_order = [0, 12, 13, 14, 9, 10, 11, 1, 2, 6, 7, 8, 3, 4, 5]
    # new_order = [0, 7, 8, 12, 13, 14, 9, 10, 11, 4, 5, 6, 1, 2, 3]
    tensor_reordered = tensor[:, new_order, :]
    return tensor_reordered

class SpecAnyskillAgent(common_agent.CommonAgent):
    def __init__(self, base_name, config):
        with open(os.path.join(os.getcwd(), config['llc_config']), 'r') as f:
            llc_config = yaml.load(f, Loader=yaml.SafeLoader)
            llc_config_params = llc_config['params']
            self._latent_dim = llc_config_params['config']['latent_dim']
        
        super().__init__(base_name, config)

        self._task_size = self.vec_env.env.task.get_task_obs_size()

        self._llc_steps = config['llc_steps']
        self._wandb_counter = config['wandb_counter']
        llc_checkpoint = config['llc_checkpoint']
        assert(llc_checkpoint != "")
        self._build_llc(llc_config_params, llc_checkpoint)

        self._latent_steps_min = 0
        # self._latent_steps_min = 32
        self._latent_steps_max = 96

        self.anyeditor = rendered.test()
        self.anyskill = anyskill.anytest()
        self.mlip_encoder = anyskill.FeatureExtractor()
        self.text_file = config['text_file']
        self.RENDER = config['render']
        self.LLM = config['llm']
        self._exp_sim = torch.zeros([32, 1024], device=self.device, dtype=torch.float32)
        self._anyskill_count = torch.zeros([self.horizon_length*5, 1024], device=self.device, dtype=torch.float32)

        self.clip_features = []
        # self.delta = torch.zeros([1024], device=self.device, dtype=torch.float32)
        # self.clip_features = torch.zeros([1, 512], device=self.device, dtype=torch.float32)
        # self.motionclip_features = torch.zeros([1, 15, 3], device=self.device, dtype=torch.float32)
        self.motionclip_features = []
        self.counter = 0
        self.headless = config['headless']
        self.motionfile = "./output/motion_feature_spec" + str(self._wandb_counter) + ".npy"
        self.imagefile = "./output/image_feature_spec" + str(self._wandb_counter) + ".npy"
        self.output = []
        return

    def env_step(self, actions, step):
        actions = self.preprocess_actions(actions)
        obs = self.obs['obs']
        self._llc_actions = torch.zeros([self._llc_steps, 1024, 28], device=self.device, dtype=torch.float32)
        anyskill_count = torch.zeros([self._llc_steps, 1024], device=self.device, dtype=torch.float32)

        rewards = 0.0
        max_anyksill = torch.zeros([1024], device=self.device, dtype=torch.float32)
        disc_rewards = 0.0
        done_count = 0.0
        terminate_count = 0.0
        for t in range(self._llc_steps): # low-level controller sample 5
            llc_actions = self._compute_llc_action(obs, actions) # get actions
            obs, aux_rewards, curr_dones, infos = self.vec_env.step(llc_actions) # 223d update actions

            if self.LLM:
                # #  ================== walk like model ==================
                # body_pos = infos['state_embeds'][:, :15, :3]  # [num, 15, 3]
                # body_rot = infos['state_embeds'][:, :15, 3:7]  # [num, 15, 4]
                # body_vel = infos['state_embeds'][:, :15, 7:10]  # [num, 15, 3]
                # body_ang_vel = infos['state_embeds'][:, :15, 10:13]  # [num, 15, 3]
                #
                # # Extract positions
                # pelvis_pos = body_pos[:, 0, :]  # pelvis
                # torso_pos = body_pos[:, 1, :]  # torso
                #
                # left_upper_arm_pos = body_pos[:, 6, :]  # left_upper_arm
                # right_upper_arm_pos = body_pos[:, 3, :]  # right_upper_arm
                #
                # left_foot_pos = body_pos[:, 14, :]  # left_foot
                # right_foot_pos = body_pos[:, 11, :]  # right_foot
                #
                # left_thigh_pos = body_pos[:, 12, :]  # left_thigh
                # left_shin_pos = body_pos[:, 13, :]  # left_shin
                #
                # right_thigh_pos = body_pos[:, 9, :]  # right_thigh
                # right_shin_pos = body_pos[:, 10, :]  # right_shin
                #
                # # 1. Reward for arms opened to the sides
                # # Vectors from torso to upper arms
                # left_upper_arm_vector = left_upper_arm_pos - torso_pos  # [num, 3]
                # right_upper_arm_vector = right_upper_arm_pos - torso_pos  # [num, 3]
                #
                # # x-axis unit vector
                # x_axis = torch.tensor([1.0, 0.0, 0.0], device=self.device).unsqueeze(0)  # [1, 3]
                #
                # # Cosine similarities with x-axis
                # left_arm_cos = torch.nn.functional.cosine_similarity(left_upper_arm_vector, -x_axis,
                #                                                      dim=1)  # Left arm to -x
                # right_arm_cos = torch.nn.functional.cosine_similarity(right_upper_arm_vector, x_axis,
                #                                                       dim=1)  # Right arm to +x
                #
                # # Temperature parameter for arms
                # arm_temperature = 0.1  # float
                #
                # # Reward for arms extended
                # reward_left_arm = torch.exp((left_arm_cos - 1.0) / arm_temperature)
                # reward_right_arm = torch.exp((right_arm_cos - 1.0) / arm_temperature)
                # reward_arms = (reward_left_arm + reward_right_arm) / 2.0  # Average reward
                #
                # # 2. Reward for standing on one leg
                # # Y positions (vertical) of feet
                # left_foot_z = left_foot_pos[:, 2]  # [num]
                # right_foot_z = right_foot_pos[:, 2]  # [num]
                #
                # # Ground level and threshold
                # ground_level = 0.0  # Assuming ground is at y=0
                # foot_on_ground_threshold = 0.05  # meters
                #
                # # Foot contact with ground
                # left_foot_on_ground = (left_foot_z < foot_on_ground_threshold)
                # right_foot_on_ground = (right_foot_z < foot_on_ground_threshold)
                #
                # # Standing on one leg mask
                # stand_on_left_leg = left_foot_on_ground & (~right_foot_on_ground)
                # stand_on_right_leg = right_foot_on_ground & (~left_foot_on_ground)
                # stand_on_one_leg = stand_on_left_leg | stand_on_right_leg
                #
                # # Reward for standing on one leg
                # reward_stand_on_one_leg = stand_on_one_leg.float()
                #
                # # 3. Reward for lifting the other leg straight up
                # # Lifted foot masks
                # lifted_left_foot = (~left_foot_on_ground)
                # lifted_right_foot = (~right_foot_on_ground)
                #
                # # Left leg vectors
                # left_thigh_vector = left_shin_pos - left_thigh_pos  # [num, 3]
                # left_shin_vector = left_foot_pos - left_shin_pos  # [num, 3]
                #
                # # Right leg vectors
                # right_thigh_vector = right_shin_pos - right_thigh_pos  # [num, 3]
                # right_shin_vector = right_foot_pos - right_shin_pos  # [num, 3]
                #
                # # Cosine similarities for leg straightness
                # cos_theta_left_leg = torch.nn.functional.cosine_similarity(left_thigh_vector, left_shin_vector,
                #                                                            dim=1)
                # cos_theta_right_leg = torch.nn.functional.cosine_similarity(right_thigh_vector, right_shin_vector,
                #                                                             dim=1)
                #
                # # Temperature parameter for leg straightness
                # leg_temperature = 0.1  # float
                #
                # # Reward for lifted leg being straight
                # reward_left_leg_straight = torch.exp(
                #     (cos_theta_left_leg - 1.0) / leg_temperature) * lifted_left_foot.float()
                # reward_right_leg_straight = torch.exp(
                #     (cos_theta_right_leg - 1.0) / leg_temperature) * lifted_right_foot.float()
                # reward_lifted_leg_straight = reward_left_leg_straight + reward_right_leg_straight
                #
                # # 4. Reward for lifting the foot high
                # # Foot lift threshold and temperature
                # foot_lift_threshold = 0.5  # meters
                # foot_lift_temperature = 0.1  # float
                #
                # # Reward for left foot lift
                # left_foot_lift_amount = left_foot_z - foot_lift_threshold  # [num]
                # reward_left_foot_lift = torch.exp(-torch.relu(
                #     foot_lift_threshold - left_foot_z) / foot_lift_temperature) * lifted_left_foot.float()
                #
                # # Reward for right foot lift
                # right_foot_lift_amount = right_foot_z - foot_lift_threshold  # [num]
                # reward_right_foot_lift = torch.exp(-torch.relu(
                #     foot_lift_threshold - right_foot_z) / foot_lift_temperature) * lifted_right_foot.float()
                #
                # # Total foot lift reward
                # reward_foot_lift = reward_left_foot_lift + reward_right_foot_lift
                #
                # # Total reward
                # llm_rewards = reward_arms + reward_stand_on_one_leg + reward_lifted_leg_straight + reward_foot_lift

                # # ==================  conduct orchestra  ==================
                # body_pos = infos['state_embeds'][:, :15, :3]  # [num_envs, 15, 3]
                # body_rot = infos['state_embeds'][:, :15, 3:7]  # [num_envs, 15, 4]
                # body_vel = infos['state_embeds'][:, :15, 7:10]  # [num_envs, 15, 3]
                # body_ang_vel = infos['state_embeds'][:, :15, 10:13]  # [num_envs, 15, 3]
                #
                # # Extract joint data
                # pelvis_pos = body_pos[:, 0, :]
                # pelvis_rot = body_rot[:, 0, :]
                # pelvis_vel = body_vel[:, 0, :]
                # torso_rot = body_rot[:, 1, :]
                # right_hand_pos = body_pos[:, 5, :]
                # right_hand_vel = body_vel[:, 5, :]
                # left_thigh_pos = body_pos[:, 12, :]
                # right_thigh_pos = body_pos[:, 9, :]
                #
                # # 1. Standing still: minimize pelvis horizontal velocity
                # pelvis_vel_reward = -torch.norm(pelvis_vel[:, :2], dim=1)
                #
                # # 2. Standing upright: encourage torso's up vector to align with world up vector
                # qx, qy, qz, qw = torso_rot[:, 0], torso_rot[:, 1], torso_rot[:, 2], torso_rot[:, 3]
                # up_z = 1 - 2 * (qx ** 2 + qy ** 2)
                # upright_reward = up_z
                #
                # # 3. Swing
                # thigh_center = (left_thigh_pos + right_thigh_pos) / 2
                # hand_to_thigh_center = right_hand_pos - thigh_center
                # hand_between_legs_reward = -torch.norm(hand_to_thigh_center[:, :2], dim=1)
                # hand_swinging_reward = torch.abs(right_hand_vel[:, 2])
                #
                # # Apply transformations with temperature parameters
                # temp_pelvis_vel = 1.0
                # temp_upright = 1.0
                # temp_hand_between_legs = 1.0
                #
                # transformed_pelvis_vel_reward = torch.exp(temp_pelvis_vel * pelvis_vel_reward)
                # transformed_upright_reward = torch.exp(temp_upright * (upright_reward - 1))
                # transformed_hand_between_legs_reward = torch.exp(temp_hand_between_legs * hand_between_legs_reward)
                #
                # # Weighting and total reward
                # w_pelvis_vel = 1.0
                # w_upright = 1.0
                # w_hand_between_legs = 1.0
                # w_hand_swinging = 1.0
                #
                # llm_rewards = (
                #         # w_pelvis_vel * transformed_pelvis_vel_reward
                #         # + w_upright * transformed_upright_reward
                #         w_hand_between_legs * transformed_hand_between_legs_reward
                #         + w_hand_swinging * hand_swinging_reward
                # )


                # # ==================  jump hurdles ==================
                body_pos = infos['state_embeds'][:, :15, :3]  # [num_envs, 15, 3]
                body_rot = infos['state_embeds'][:, :15, 3:7]  # [num_envs, 15, 4]
                body_vel = infos['state_embeds'][:, :15, 7:10]  # [num_envs, 15, 3]
                body_ang_vel = infos['state_embeds'][:, :15, 10:13]  # [num_envs, 15, 3]

                # Extract joint data
                pelvis_pos = body_pos[:, 0, :]
                pelvis_rot = body_rot[:, 0, :]
                pelvis_vel = body_vel[:, 0, :]
                torso_rot = body_rot[:, 1, :]
                right_hand_pos = body_pos[:, 5, :]
                right_hand_vel = body_vel[:, 5, :]
                left_thigh_pos = body_pos[:, 12, :]
                right_thigh_pos = body_pos[:, 9, :]

                # 1. Standing still: minimize pelvis horizontal velocity
                pelvis_vel_reward = -torch.norm(pelvis_vel[:, :2], dim=1)

                # 2. Body leaning forward: encourage torso to lean forward
                qx, qy, qz, qw = torso_rot[:, 0], torso_rot[:, 1], torso_rot[:, 2], torso_rot[:, 3]
                up_z = 1 - 2 * (qx ** 2 + qy ** 2)  # Cosine of the angle between torso up and world up
                desired_pitch = torch.tensor(0.5)  # Desired pitch angle in radians (e.g., ~28.6 degrees)
                desired_up_z = torch.cos(desired_pitch)
                temp_posture = 0.5  # Temperature parameter for posture reward
                posture_reward = torch.exp(-((up_z - desired_up_z) ** 2) / temp_posture)

                # 3. Swing kettlebell between legs
                thigh_center = (left_thigh_pos + right_thigh_pos) / 2
                hand_to_thigh_center = right_hand_pos - thigh_center
                hand_between_legs_reward = -torch.norm(hand_to_thigh_center[:, :2], dim=1)
                hand_swinging_reward = torch.abs(right_hand_vel[:, 2])

                # Apply transformations with temperature parameters
                temp_pelvis_vel = 1.0
                temp_hand_between_legs = 1.0

                transformed_pelvis_vel_reward = torch.exp(temp_pelvis_vel * pelvis_vel_reward)
                transformed_hand_between_legs_reward = torch.exp(temp_hand_between_legs * hand_between_legs_reward)

                # Weighting and total reward
                w_pelvis_vel = 0.0
                w_posture = 1.0
                w_hand_between_legs = 1.0
                w_hand_swinging = 1.0

                llm_rewards = (
                        w_pelvis_vel * transformed_pelvis_vel_reward
                        + w_posture * posture_reward
                        + w_hand_between_legs * transformed_hand_between_legs_reward
                        + w_hand_swinging * hand_swinging_reward
                )


            if self.RENDER:
                if self.headless == False:
                    images = self.vec_env.env.task.render_img()
                else:
                    # print("apply the headless mode")
                    images = self.vec_env.env.task.render_headless()

                image_features = self.mlip_encoder.encode_images(images)
                state_embeds = infos['state_embeds'][:, :15, :3]
                # print("we have render")
                self.clip_features.append(image_features.data.cpu().numpy())
                self.motionclip_features.append(state_embeds.data.cpu().numpy())
                image_features_norm = image_features / image_features.norm(dim=-1, keepdim=True)

            else:
                quat_embeds = infos['state_embeds'][:, :15, 3:7]
                root_rot_quat = quat_embeds[:, 0, :].cpu()
                root_euler = R.from_quat(root_rot_quat).as_euler('xyz', degrees=False)

                root_euler = root_euler[:, [1, 0, 2]]

                test = infos['dof_pos']
                body_rot_euler = test.view(1024, 28)

                input_euler = transform_tensor(body_rot_euler, root_euler)
                input_euler_zxy = input_euler[:, :, [1, 2, 0]]
                input_euler_zxy_reorg = reorg(input_euler_zxy.view(1024, 15, 3))


                # angle_embeds = quat_to_angle(quat_embeds.view(-1, 4))
                # angle_embeds = reorg(angle_embeds.view(1024, 15, 3))
                # self.output.append(input_euler_zxy_reorg[0,:,:].data.cpu().numpy())
                # np.save('test.npy', self.output)

                image_features_mlp = self.anyeditor.get_motion_embedding(input_euler_zxy_reorg.view(1024, -1))
                image_features_norm = image_features_mlp / image_features_mlp.norm(dim=-1, keepdim=True)
                # print("we have MLP")

            # eu_dis = F.pairwise_distance(image_features_norm, image_features_mlp_norm, keepdim=True)
            # cos_ids = F.cosine_similarity(image_features_norm, image_features_mlp_norm, dim=1)

            done_count += curr_dones
            terminate_count += infos['terminate']
            amp_obs = infos['amp_obs']
            curr_disc_reward = self._calc_disc_reward(amp_obs)
            disc_rewards += curr_disc_reward
            self._llc_actions[t] = llc_actions

            # average
            # if not self.LLM:
            anyskill_rewards, delta, similarity = self.vec_env.env.task.compute_anyskill_reward(image_features_norm, self._text_latents,
                                                                             self._latent_text_idx)


            # # max
            # max_anyksill = torch.max(max_anyksill, anyskill_rewards)
            # curr_rewards = max_anyksill

            # together
            # curr_rewards = llm_rewards

            if self.LLM:
                curr_rewards = llm_rewards + anyskill_rewards
            else:
                curr_rewards = anyskill_rewards


            # only model
            # curr_rewards = anyskill_rewards + eureka_rewards

            # # only render
            # curr_rewards = anyskill_rewards


            # velocity
            # curr_rewards = anyskill_rewards + aux_rewards #(1024,)
            # anyskill_count[t] = anyskill_rewards #(5, 1024)

            # # sliding window
            # idx = 5 * step + t
            # self._anyskill_count[idx] = anyskill_rewards
            # exp = torch.mean(anyskill_count[:8, :], dim=0)
            # curr_rewards -= exp

            rewards += curr_rewards

        # self._exp_sim[step] = anyskill_count.mean(dim=0) #(1024,)
        rewards /= self._llc_steps #(1024,)
        disc_rewards /= self._llc_steps
        dones = torch.zeros_like(done_count)
        dones[done_count > 0] = 1.0
        terminate = torch.zeros_like(terminate_count)
        terminate[terminate_count > 0] = 1.0
        infos['terminate'] = terminate
        infos['disc_rewards'] = disc_rewards

        # # max
        #     anyskill_rewards, similarity = self.vec_env.env.task.compute_anyskill_reward(image_features_norm, self._text_latents,
        #                                                                      self._latent_text_idx)
        #     curr_rewards = anyskill_rewards
        #     rewards = torch.max(curr_rewards)
        #
        # # self._exp_sim[step] = anyskill_count.mean(dim=0) #(1024,)
        # # rewards /= self._llc_steps #(1024,)
        # disc_rewards /= self._llc_steps
        # dones = torch.zeros_like(done_count)
        # dones[done_count > 0] = 1.0
        # terminate = torch.zeros_like(terminate_count)
        # terminate[terminate_count > 0] = 1.0
        # infos['terminate'] = terminate
        # infos['disc_rewards'] = disc_rewards


        # wandb.log({"info/delta": delta.mean().item()}, step)
        # wandb.log({"reward/llm_reward": llm_rewards.mean().item()}, step)
        wandb.log({"reward/spec_anyskill_reward": anyskill_rewards.mean().item()}, step)
        wandb.log({"reward/spec_aux_reward": aux_rewards.mean().item()}, step)
        wandb.log({"info/eposide": self.vec_env.env.task.eposide}, step)
        # wandb.log({"info/clip_similarity": similarity.mean().item()}, step)
        # wandb.log({"info/eu_dis": eu_dis.mean().item()}, step)
        # wandb.log({"info/cos_dis": cos_ids.mean().item()}, step)
        wandb.log({"info/step": step}, step)
        self.vec_env.env.task.eposide = 0

        if self.is_tensor_obses:
            if self.value_size == 1:
                rewards = rewards.unsqueeze(1)
            return self.obs_to_tensors(obs), rewards.to(self.ppo_device), dones.to(self.ppo_device), infos, self._llc_actions
        else:
            if self.value_size == 1:
                rewards = np.expand_dims(rewards, axis=1)
            return self.obs_to_tensors(obs), torch.from_numpy(rewards).to(self.ppo_device).float(), torch.from_numpy(dones).to(self.ppo_device), infos, self._llc_actions

    def cast_obs(self, obs):
        obs = super().cast_obs(obs)
        self._llc_agent.is_tensor_obses = self.is_tensor_obses
        return obs

    def preprocess_actions(self, actions):
        clamped_actions = torch.clamp(actions, -1.0, 1.0)
        if not self.is_tensor_obses:
            clamped_actions = clamped_actions.cpu().numpy()
        return clamped_actions

    def _reset_latents(self, env_ids):
        n = len(env_ids)
        z, z_text_idx = self._sample_latents(n)
        self._text_latents[env_ids] = z
        self._latent_text_idx[env_ids] = z_text_idx

        if (self.vec_env.env.task.viewer):
            self._change_char_color(env_ids)

        return

    def _sample_latents(self, n):
        z, z_text_idx = self.model.a2c_network.sample_text_embeddings(n, self.text_features, self.text_weights)
        return z, z_text_idx

    def _update_latents(self):
        new_latent_envs = self._latent_reset_steps <= self.vec_env.env.task.progress_buf

        need_update = torch.any(new_latent_envs)
        if (need_update):
            new_latent_env_ids = new_latent_envs.nonzero(as_tuple=False).flatten()
            self._reset_latents(new_latent_env_ids)
            self._latent_reset_steps[new_latent_env_ids] += torch.randint_like(self._latent_reset_steps[new_latent_env_ids],
                                                                               low=self._latent_steps_min,
                                                                               high=self._latent_steps_max)
            if (self.vec_env.env.task.viewer):
                self._change_char_color(new_latent_env_ids)

        return

    def play_steps(self):
        self.counter += 1
        self.set_eval()
        '''
            s, a<-z, r<-(s:img, z )
        '''
        epinfos = []
        done_indices = []
        update_list = self.update_list

        for n in range(self.horizon_length):
            self._update_latents()
            self.obs = self.env_reset(done_indices)
            self.experience_buffer.update_data('obses', n, self.obs['obs'])

            res_dict = self.get_action_values(self.obs)

            for k in update_list:
                self.experience_buffer.update_data(k, n, res_dict[k]) 

            if self.has_central_value:
                self.experience_buffer.update_data('states', n, self.obs['states'])

            # if n < 2 and n > 23:
            #     continue
            self.obs, rewards, self.dones, infos, _llc_actions = self.env_step(res_dict['actions'], n)

            # # Calculate RCLIP score
            # rewards -= self._exp_sim.mean(dim=0)  # (32,1024)
            shaped_rewards = self.rewards_shaper(rewards)

            self.experience_buffer.update_data('rewards', n, shaped_rewards)
            self.experience_buffer.update_data('next_obses', n, self.obs['obs'])
            self.experience_buffer.update_data('dones', n, self.dones)

            self.experience_buffer.update_data('disc_rewards', n, infos['disc_rewards'])

            style_rewards = self._calc_style_reward(res_dict['actions'])
            self.experience_buffer.update_data('style_rewards', n, style_rewards)

            terminated = infos['terminate'].float()
            terminated = terminated.unsqueeze(-1)
            next_vals = self._eval_critic(self.obs)
            next_vals *= (1.0 - terminated)
            self.experience_buffer.update_data('next_values', n, next_vals)

            self.current_rewards += rewards
            self.current_lengths += 1
            all_done_indices = self.dones.nonzero(as_tuple=False)
            done_indices = all_done_indices[::self.num_agents]
  
            self.game_rewards.update(self.current_rewards[done_indices])
            self.game_lengths.update(self.current_lengths[done_indices])
            self.algo_observer.process_infos(infos, done_indices)

            not_dones = 1.0 - self.dones.float()

            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones

            done_indices = done_indices[:, 0]

        mb_fdones = self.experience_buffer.tensor_dict['dones'].float()
        mb_values = self.experience_buffer.tensor_dict['values']
        mb_next_values = self.experience_buffer.tensor_dict['next_values']

        mb_rewards = self.experience_buffer.tensor_dict['rewards']
        mb_disc_rewards = self.experience_buffer.tensor_dict['disc_rewards']
        mb_style_rewards = self.experience_buffer.tensor_dict['style_rewards']
        # mb_text_latents = self.experience_buffer.tensor_dict['text_latents']

        mb_rewards = self._combine_rewards(mb_rewards, mb_disc_rewards, mb_style_rewards)

        mb_advs = self.discount_values(mb_fdones, mb_values, mb_rewards, mb_next_values)
        mb_returns = mb_advs + mb_values

        batch_dict = self.experience_buffer.get_transformed_list(a2c_common.swap_and_flatten01, self.tensor_list)
        batch_dict['returns'] = a2c_common.swap_and_flatten01(mb_returns)
        batch_dict['played_frames'] = self.batch_size

        # if self.counter % 150 == 1:
        #
        #     np.save(self.motionfile, self.motionclip_features)
        #     np.save(self.imagefile, self.clip_features)
        #
        # print("we have run {} steps and save data".format(self.counter))

        return batch_dict
    
    def _load_config_params(self, config):
        super()._load_config_params(config)
        
        self._task_reward_w = config['task_reward_w']
        self._disc_reward_w = config['disc_reward_w']
        self._style_reward_w = config['style_reward_w']
        return

    def _get_mean_rewards(self):
        rewards = super()._get_mean_rewards()
        rewards *= self._llc_steps
        return rewards

    def _setup_action_space(self):
        super()._setup_action_space()
        self.actions_num = self._latent_dim
        return

    def init_tensors(self):
        super().init_tensors()

        del self.experience_buffer.tensor_dict['actions']
        del self.experience_buffer.tensor_dict['mus']
        del self.experience_buffer.tensor_dict['sigmas']

        batch_shape = self.experience_buffer.obs_base_shape
        # self.experience_buffer.tensor_dict['text_latents'] = torch.zeros(batch_shape + (512,),
        #                                                             dtype=torch.float32, device=self.ppo_device)
        self.experience_buffer.tensor_dict['actions'] = torch.zeros(batch_shape + (self._latent_dim,),
                                                                    dtype=torch.float32, device=self.ppo_device)
        self.experience_buffer.tensor_dict['mus'] = torch.zeros(batch_shape + (self._latent_dim,),
                                                                dtype=torch.float32, device=self.ppo_device)
        self.experience_buffer.tensor_dict['sigmas'] = torch.zeros(batch_shape + (self._latent_dim,),
                                                                   dtype=torch.float32, device=self.ppo_device)
        
        self.experience_buffer.tensor_dict['disc_rewards'] = torch.zeros_like(self.experience_buffer.tensor_dict['rewards'])
        self.experience_buffer.tensor_dict['style_rewards'] = torch.zeros_like(self.experience_buffer.tensor_dict['rewards'])
        self.tensor_list += ['disc_rewards', 'style_rewards']

        batch_shape = self.experience_buffer.obs_base_shape
        self._latent_reset_steps = torch.zeros(batch_shape[-1], dtype=torch.int32, device=self.ppo_device)

        texts, texts_weights = load_texts(self.text_file)
        self.text_features = self.mlip_encoder.encode_texts(texts)
        self.text_weights = torch.tensor(texts_weights, device=self.device)
        self._text_latents = torch.zeros((batch_shape[-1], 512), dtype=torch.float32,
                                         device=self.ppo_device)
        self._latent_text_idx = torch.zeros((batch_shape[-1],), dtype=torch.long, device=self.ppo_device)

        return

    def _get_motion_encoding(self):
        all_encoded_demo_amp_obs = []
        for motion_id in range(self.vec_env.env.task._motion_lib._motion_weights.shape[0]):
            motion_amp_obs = self.vec_env.env.task.fetch_amp_obs_demo_per_id(3, motion_id)[-1].view(3,
                                                                                                     self.vec_env.env.task._num_amp_obs_enc_steps,
                                                                                                     self.vec_env.env.task._num_amp_obs_per_step)
            preproc_amp_obs = self._llc_agent._preproc_amp_obs(motion_amp_obs) #(32,60,125)
            encoded_demo_amp_obs = self._llc_agent.model.a2c_network.eval_enc(preproc_amp_obs)
            all_encoded_demo_amp_obs.append(encoded_demo_amp_obs)
        stacked_obs = torch.stack(all_encoded_demo_amp_obs, dim=0).view(-1, 64)
        sample_indices = torch.randperm(stacked_obs.size(0))[:128]
        all_encoded_demo_amp_obs = stacked_obs[sample_indices].unsqueeze(0) #(1,128,64)

        return all_encoded_demo_amp_obs


    def _build_llc(self, config_params, checkpoint_file):
        network_params = config_params['network']

        network_builder = calm_network_builder.CALMBuilder()

        network_builder.load(network_params)

        network = calm_models.ModelCALMContinuous(network_builder)

        llc_agent_config = self._build_llc_agent_config(config_params, network)

        self._llc_agent = calm_agent.CALMAgent('llc', llc_agent_config)

        self._llc_agent.restore(checkpoint_file)
        print("Loaded LLC checkpoint from {:s}".format(checkpoint_file))
        self._llc_agent.set_eval()

        # # Before
        # enc_amp_obs = self._llc_agent._fetch_amp_obs_demo(128) #(128,7500)
        # if len(enc_amp_obs) == 2:
        #     enc_amp_obs = enc_amp_obs[0]
        #
        # preproc_enc_amp_obs = self._llc_agent._preproc_amp_obs(enc_amp_obs)
        # self.encoded_motion_old = self._llc_agent.model.a2c_network.eval_enc(amp_obs=preproc_enc_amp_obs).unsqueeze(0) #(1,128,64)

        # After
        self.encoded_motion = self._get_motion_encoding()

        return

    def _build_llc_agent_config(self, config_params, network):
        llc_env_info = copy.deepcopy(self.env_info)
        obs_space = llc_env_info['observation_space']
        obs_size = obs_space.shape[0]
        obs_size -= self._task_size
        llc_env_info['observation_space'] = spaces.Box(obs_space.low[:obs_size], obs_space.high[:obs_size])

        config = config_params['config']
        config['network'] = network
        config['num_actors'] = self.num_actors
        config['features'] = {'observer': self.algo_observer}
        config['env_info'] = llc_env_info
        config['minibatch_size'] = 1
        config['amp_batch_size'] = 32
        config['amp_minibatch_size'] = 1
        config['enable_eps_greedy'] = False
        config['vec_env'] = self.vec_env

        return config

    def _compute_llc_action(self, obs, actions):
        llc_obs = self._extract_llc_obs(obs)
        processed_obs = self._llc_agent._preproc_obs(llc_obs)

        z = torch.nn.functional.normalize(actions, dim=-1)
        mu, _ = self._llc_agent.model.a2c_network.eval_actor(processed_obs, z)
        llc_action = mu
        llc_action = self._llc_agent.preprocess_actions(llc_action)

        return llc_action

    def _extract_llc_obs(self, obs):
        obs_size = obs.shape[-1]
        llc_obs = obs[..., :obs_size - self._task_size]
        return llc_obs

    def _calc_disc_reward(self, amp_obs):
        disc_reward = self._llc_agent._calc_disc_rewards(amp_obs)
        return disc_reward

    def _calc_style_reward(self, action):
        z = torch.nn.functional.normalize(action, dim=-1)
        style_reward = torch.max((F.cosine_similarity(z.unsqueeze(1), self.encoded_motion, dim=-1) + 1) / 2, dim=1)[0]
        # cal all similarity, weighted average (sim)

        return style_reward.unsqueeze(-1)

    # # After
    # def _calc_style_reward(self, action):
    #     requested_heights = self.vec_env.env.task._tar_locomotion_index.view(-1)
    #     style_reward = torch.zeros((action.shape[0]), device=self.ppo_device, dtype=torch.float32)
    #     z = torch.nn.functional.normalize(action, dim=-1)
    #
    #     for motion_type in range(3):
    #         motion_mask = requested_heights == motion_type
    #         style_reward[motion_mask] = torch.max((F.cosine_similarity(z[motion_mask].view(-1, 1, self._latent_dim), self.encoded_motion[motion_type].view(1, -1, self._latent_dim), dim=-1) + 1) / 2, dim=1)[0]
    #
    #     return style_reward.unsqueeze(-1)

    def _combine_rewards(self, task_rewards, disc_rewards, style_rewards):
        combined_rewards = self._task_reward_w * task_rewards + \
                         + self._disc_reward_w * disc_rewards + self._style_reward_w * style_rewards

        return combined_rewards

    def _record_train_batch_info(self, batch_dict, train_info):
        super()._record_train_batch_info(batch_dict, train_info)
        train_info['disc_rewards'] = batch_dict['disc_rewards']
        train_info['style_rewards'] = batch_dict['style_rewards']
        return

    def _log_train_info(self, train_info, frame):
        super()._log_train_info(train_info, frame)

        disc_reward_std, disc_reward_mean = torch.std_mean(train_info['disc_rewards'])
        self.writer.add_scalar('info/disc_reward_mean', disc_reward_mean.item(), frame)
        self.writer.add_scalar('info/disc_reward_std', disc_reward_std.item(), frame)

        style_reward_std, style_reward_mean = torch.std_mean(train_info['style_rewards'])
        self.writer.add_scalar('info/style_reward_mean', style_reward_mean.item(), frame)
        self.writer.add_scalar('info/style_reward_std', style_reward_std.item(), frame)
        return


    def _change_char_color(self, env_ids):
        base_col = np.array([0.4, 0.4, 0.4])
        range_col = np.array([0.0706, 0.149, 0.2863])
        range_sum = np.linalg.norm(range_col)

        rand_col = np.random.uniform(0.0, 1.0, size=3)
        rand_col = range_sum * rand_col / np.linalg.norm(rand_col)
        rand_col += base_col
        self.vec_env.env.task.set_char_color(rand_col, env_ids)
        return

def load_texts(text_file):
    ext = os.path.splitext(text_file)[1]
    assert ext == ".yaml"
    weights = []
    texts = []
    with open(os.path.join(os.getcwd(), text_file), 'r') as f:
        text_config = yaml.load(f, Loader=yaml.SafeLoader)

    text_list = text_config['texts']
    for text_entry in text_list:
        curr_text = text_entry['text']
        curr_weight = text_entry['weight']
        assert(curr_weight >= 0)
        weights.append(curr_weight)
        texts.append(curr_text)
    return texts, weights