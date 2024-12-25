# Aim:     Extract image features from rendered images using the CLIP model for further analysis or training.
# Method:  Initialize the CLIP model, preprocess images from specified viewpoints, and extract their features. 
#          The features are then saved to joblib files for later use.
# Modules: os, numpy, torch, PIL (Image), open_clip, joblib
# Args:    render_directory (str): Directory containing images for feature extraction.
#          output_file (str): Directory to save the extracted features in joblib format.

import os
import numpy as np
import torch
from PIL import Image
import open_clip
from tqdm import tqdm
from joblib import dump
import logging
import argparse

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ImageFeatureExtractor:
    """Class to extract image features using the CLIP model."""
    
    def __init__(self, model_name='ViT-B-32', pretrained='laion2b_s34b_b79k'):
        """Initialize the ImageFeatureExtractor with the specified model."""
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        logging.info("Initialized ImageFeatureExtractor.")

    def extract_features(self, viewpoint, render_directory):
        """Extract image features for a given viewpoint."""
        root_path = os.path.join(os.getcwd(), render_directory, viewpoint)
        img_nums = len(os.listdir(root_path))
        img_feature_array = np.empty((img_nums + 1, 512), dtype=float)
        
        for i in tqdm(range(img_nums + 1), desc=f'Processing {viewpoint}'):
            img_path = os.path.join(root_path, f'{i}.png')
            if not os.path.exists(img_path):
                logging.warning(f"Image not found: {img_path}")
                continue
            
            img = self.preprocess(Image.open(img_path)).unsqueeze(0)
            img = img.to(self.device)
            with torch.no_grad():
                img_feature = self.model.encode_image(img)
            img_feature_array[i] = img_feature.cpu().numpy()

        return img_feature_array

def main(render_directory, output_file):
    """Main function to start feature extraction."""
    extractor = ImageFeatureExtractor()
    viewpoints = ['front', 'side', 'oblique', 'rear_side', 'rear']
    for viewpoint in viewpoints:
        features = extractor.extract_features(viewpoint, render_directory)
        output_path = os.path.join(output_file, f'{viewpoint}.joblib')
        dump(features, output_path)
        logging.info(f"Saved features for viewpoint {viewpoint} to {output_path}.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract image features using CLIP.")
    parser.add_argument('--render_directory', type=str, default='data/rendered_images/idea400/', help='Directory containing images for extraction.')
    parser.add_argument('--output_file', type=str, default='data/img_feature/img_feature_idea400/', help='Directory to save the extracted features.')
    args = parser.parse_args()
    main(args.render_directory, args.output_file)
