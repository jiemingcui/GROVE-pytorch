# Aim:     Train a neural network model to map pose data to corresponding image features for pose recognition tasks.
# Method:  Load pose and image feature data, preprocess them, and train a multi-layer perceptron (MLP) model using 
#          weighted sampling for imbalanced classes.
# Modules: os, numpy, joblib, torch
# Args:    pose_data_amass (str): Path to AMASS pose data.
#          pose_data_idea400 (str): Path to IDEA400 pose data.
#          pose_data_failure_cases (str): Path to failure cases data.
#          pose_data_other_motionx (str): Path to other motion data.
#          img_feature_path (str): Base path for image features.
#          cluster_dict (str): Path to cluster indices.
#          use_wandb (bool): Initialize wandb for logging.
#          model_save_path (str): Path to save the trained model.

import os
import numpy as np
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm
import wandb
import logging
import argparse

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration class
class Config:
    def __init__(self):
        self.num_epochs = 2
        self.batch_size = 512
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.calm_joints_order = [
            "pelvis", "left_hip", "left_knee", "left_foot", 
            "right_hip", "right_knee", "right_foot", "spine1", 
            "head", "left_shoulder", "left_elbow", "left_pinky1", 
            "right_shoulder", "right_elbow", "right_pinky1"
        ]
        self.smplx_joints_order = [
            "pelvis", "left_hip", "right_hip", "spine1", 
            "left_knee", "right_knee", "spine2", "left_ankle", 
            "right_ankle", "spine3", "left_foot", "right_foot", 
            "neck", "left_collar", "right_collar", "head", 
            "left_shoulder", "right_shoulder", "left_elbow", 
            "right_elbow", "left_wrist", "right_wrist", 
            "left_pinky1", "right_pinky1"
        ]
        self.vp_list = ['front', 'side', 'oblique', 'rear_side', 'rear']
        self.input_dim = 45
        self.hidden_dims = [256, 1024]
        self.output_dim = 512

# Argument parser
def parse_args():
    parser = argparse.ArgumentParser(description="Pose to Clip Training")
    parser.add_argument("--pose_data_amass", type=str, default="data/pose_data/pose_data_amass.joblib", help="Path to AMASS pose data")
    parser.add_argument("--pose_data_idea400", type=str, default="data/pose_data/pose_data_idea400.joblib", help="Path to IDEA400 pose data")
    parser.add_argument("--pose_data_failure_cases", type=str, default="data/pose_data/failure_cases.npy", help="Path to failure cases data")
    parser.add_argument("--pose_data_other_motionx", type=str, default="data/pose_data/pose_data_other_motionx.npy", help="Path to other motion data")
    parser.add_argument("--img_feature_path", type=str, default="data/img_feature/img_feature_", help="Base path for image features")
    parser.add_argument("--cluster_dict", type=str, default="classify/indices.joblib", help="Path to cluster indices")
    parser.add_argument("--use_wandb", action='store_true', help="Initialize wandb for logging")
    parser.add_argument("--model_save_path", type=str, default="models/pose2clip.pth", help="Path to save the trained model")
    return parser.parse_args()


# Custom Dataset class
class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# MLP Model class
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        in_dim = input_dim
        for hidden_dim in hidden_dims:
            self.layers.append(nn.Linear(in_dim, hidden_dim))
            self.layers.append(nn.ReLU())
            in_dim = hidden_dim
        self.output = nn.Linear(in_dim, output_dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.output(x)

def xavier_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)

# Training class
class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, config, use_wandb, model_save_path):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.config = config
        self.use_wandb = use_wandb
        self.model_save_path = model_save_path  # Add this line

    def train(self):
        for epoch in range(self.config.num_epochs):
            self.model.train()
            train_pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc=f"Epoch {epoch + 1}/{self.config.num_epochs}")
            for i, (data, target) in train_pbar:
                data, target = data.to(self.config.device), target.to(self.config.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                train_pbar.set_postfix(loss=loss.item())
                if self.use_wandb:
                    wandb.log({"train_loss": loss.item()})

            self.validate(epoch)

            # Save model
            torch.save(self.model.state_dict(), self.model_save_path)  # Use the new path
            if self.use_wandb:
                wandb.save(self.model_save_path)

    def validate(self, epoch):
        self.model.eval()
        val_pbar = tqdm(enumerate(self.val_loader), total=len(self.val_loader), desc=f"Validation Epoch {epoch + 1}")
        val_loss = 0
        with torch.no_grad():
            for i, (data, target) in val_pbar:
                data, target = data.to(self.config.device), target.to(self.config.device)
                output = self.model(data)
                val_loss += self.criterion(output, target).item()
            val_loss /= len(self.val_loader)
            if self.use_wandb:
                wandb.log({"val_loss": val_loss})
            logger.info(f'Epoch {epoch}, Validation Loss: {val_loss}')

def main():
    args = parse_args()
    config = Config()

    cwd = os.getcwd()

    logger.info("Loading pose data...")
    pose_data_amass = joblib.load(args.pose_data_amass).astype(np.float32)
    logger.info(f"Pose Data AMASS's Shape: {pose_data_amass.shape}")
    pose_data_idea400 = joblib.load(args.pose_data_idea400).astype(np.float32)
    logger.info(f"Pose Data IDEA400's Shape: {pose_data_idea400.shape}")
    pose_data_failure_cases = np.load(args.pose_data_failure_cases).astype(np.float32)
    logger.info(f"Pose Data Failure Cases's Shape: {pose_data_failure_cases.shape}")
    pose_data_other_motionx = np.load(args.pose_data_other_motionx).astype(np.float32)
    logger.info(f"Pose Data Other MotionX's Shape: {pose_data_other_motionx.shape}")

    smplx2calm = [3 * config.smplx_joints_order.index(joint) for joint in config.calm_joints_order]
    pose_data = np.concatenate((pose_data_amass, pose_data_idea400, pose_data_failure_cases, pose_data_other_motionx) * 5, axis=0)
    
    calm_pose_data = np.zeros((len(pose_data), 45))
    for i, index in enumerate(smplx2calm):
        calm_pose_data[:, 3*i:3*i+3] = pose_data[:, index:index+3]
    pose_data = calm_pose_data.astype(np.float32)
    logger.info(f"Pose Data's Shape: {pose_data.shape}")

    img_features_all = []
    for vp in config.vp_list:
        for dataset in ['amass', 'idea400', 'failure_cases', 'other_motionx']:
            img_features = joblib.load(f"{args.img_feature_path}{dataset}/{vp}.joblib").astype(np.float32)
            img_features_all.append(img_features)
            logger.info(f"{dataset.capitalize()}'s {vp} Image Features' Shape: {img_features.shape}")

    img_features = np.concatenate(img_features_all, axis=0)
    logger.info(f"Image Features' Shape: {img_features.shape}")

    cluster_dict = joblib.load(args.cluster_dict)

    # Get weights
    class_counts = {key: len(value) for key, value in cluster_dict.items()}
    weights_per_class = {key: 0.002 / count for key, count in class_counts.items()}
    weights = np.zeros(len(pose_data) // 5)
    
    for key, value in cluster_dict.items():
        class_weight = weights_per_class[key]
        for index in value:
            weights[index] = class_weight
            
    weights = np.concatenate((weights,) * 5, axis=0)
    weights = torch.FloatTensor(weights / 5).to(config.device)
    
    # Load data
    dataset = CustomDataset(pose_data, img_features)
    logger.info("Dataset has been loaded")

    # Initialize model
    model = MLP(input_dim=config.input_dim, hidden_dims=config.hidden_dims, output_dim=config.output_dim).to(config.device)
    model.apply(xavier_init)

    # Initialize criterion and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())

    # Split data
    split_index = int(len(pose_data) * 0.8)
    train_idx = np.arange(split_index)
    val_idx = np.arange(split_index, len(pose_data))

    # Initialize wandb if required
    if args.use_wandb:
        wandb.init(project="CLIP distillation", name=f"pose2clip", config=config.__dict__)

    # Initialize sampler for training and validation
    train_sampler = WeightedRandomSampler(weights[train_idx], len(train_idx))
    val_sampler = WeightedRandomSampler(weights[val_idx], len(val_idx))

    # Initialize data loader
    train_loader = DataLoader(dataset, batch_size=config.batch_size, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=config.batch_size, sampler=val_sampler)

    # Create Trainer and start training
    trainer = Trainer(model, train_loader, val_loader, criterion, optimizer, config, use_wandb=args.use_wandb, model_save_path=args.model_save_path)
    trainer.train()

    if args.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()
