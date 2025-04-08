#!/usr/bin/env python
# AlgonautsDS Feature Extraction using DeepJuice
# Based on: https://www.kaggle.com/code/ccc3ready/feauture-extraction-latest/
# Dataset: https://www.kaggle.com/datasets/ckadirt/algonauts2025nsl

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import h5py
import argparse
import sys
from pathlib import Path
import requests

# Add path to DeepJuice
sys.path.append('/home/DeepJuiceDev')

# Import DeepJuice modules
from deepjuice import *
from deepjuice.model_zoo import get_deepjuice_model, get_model_options
from deepjuice.systemops.devices import count_cuda_devices

# Configure arguments
parser = argparse.ArgumentParser(description='Extract features from Algonauts dataset using DeepJuice')
parser.add_argument('--data-dir', type=str, default='algonauts_data', help='Directory to store/load dataset')
parser.add_argument('--output-dir', type=str, default='extracted_features', help='Directory to save extracted features')
parser.add_argument('--model-name', type=str, default='vit_large_patch16_224', help='Model to use for feature extraction')
parser.add_argument('--models', type=str, nargs='+', help='Multiple models to use for feature extraction (overrides --model-name)')
parser.add_argument('--batch-size', type=int, default=32, help='Batch size for feature extraction')
parser.add_argument('--frame-sampling-rate', type=int, default=1, help='Sample 1 out of every N frames (default: 1 = use all frames)')
parser.add_argument('--sampling-rates', type=int, nargs='+', help='Multiple sampling rates to try (overrides --frame-sampling-rate)')
parser.add_argument('--use-datalad', action='store_true', help='Use Datalad to download the dataset')
parser.add_argument('--upload-hf', action='store_true', help='Upload features to HuggingFace')
parser.add_argument('--hf-token', type=str, help='HuggingFace API token')
parser.add_argument('--list-models', action='store_true', help='List available models in DeepJuice model zoo')
args = parser.parse_args()

# Make the original import modules optional # JCT 20250408
if not args.data_dir
    if args.use_datalad:
        from datalad.api import clone, get
    else:
        import kagglehub

# Set device
device = torch.device('cuda' if torch.cuda.is_available() and count_cuda_devices() > 0 else 'cpu')
print(f"Using device: {device}")

# Create output directory
os.makedirs(args.output_dir, exist_ok=True)

class AlgonautsDataset(Dataset):
    def __init__(self, data_dir, transform=None, frame_sampling_rate=1):
        """
        Args:
            data_dir: Directory containing the Algonauts dataset
            transform: Image transformations to apply
            frame_sampling_rate: Sample 1 out of every N frames (default: 1 = use all frames)
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.frame_sampling_rate = frame_sampling_rate
        self.frame_paths = self._get_frame_paths()
        
    def _get_frame_paths(self):
        frame_files = []
        for subject_dir in sorted(self.data_dir.glob('sub-*')):
            if not subject_dir.is_dir():
                continue
                
            for task_dir in sorted(subject_dir.glob('ses-task*')):
                if not task_dir.is_dir():
                    continue
                    
                for run_dir in sorted(task_dir.glob('func/run*')):
                    if not run_dir.is_dir():
                        continue
                        
                    frames_dir = run_dir / 'frames'
                    if not frames_dir.is_dir():
                        continue
                    
                    # Get all frames, sort them, and apply sampling rate    
                    all_frames = sorted(frames_dir.glob('*.jpg'))
                    sampled_frames = all_frames[::self.frame_sampling_rate]
                    frame_files.extend(sampled_frames)
                        
        return frame_files
    
    def __len__(self):
        return len(self.frame_paths)
    
    def __getitem__(self, idx):
        frame_path = self.frame_paths[idx]
        
        # Extract metadata
        parts = frame_path.parts
        sub_idx = parts.index(next(p for p in parts if p.startswith('sub-')))
        subject = parts[sub_idx]
        session = parts[sub_idx + 1]
        run = parts[sub_idx + 3]
        frame_name = frame_path.name
        
        # Load image
        img = Image.open(frame_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
            
        return {
            'image': img,
            'metadata': {
                'subject': subject,
                'session': session,
                'run': run,
                'frame': frame_name,
                'path': str(frame_path)
            }
        }

def download_dataset(data_dir, use_datalad=False):
    """Download the Algonauts dataset using Datalad or from Kaggle"""
    if use_datalad:
        print("Downloading dataset using Datalad...")
        # Clone the dataset repository
        clone(source="https://github.com/courtois-neuromod/algonauts_2025.competitors", 
              path=data_dir)
        # Get the data
        os.chdir(data_dir)
        get(path="*")
        os.chdir("..")
    else:
        print("Downloading dataset from Kaggle using kagglehub...")
        # Download latest version of the dataset
        kaggle_path = kagglehub.dataset_download("ckadirt/algonauts2025nsl")
        print(f"Dataset downloaded to: {kaggle_path}")
        
        # Create symlink or copy to the specified data directory
        os.makedirs(data_dir, exist_ok=True)
        
        # Create symbolic links from kaggle path to data_dir
        for item in Path(kaggle_path).glob('*'):
            target = Path(data_dir) / item.name
            if not target.exists():
                if os.name == 'nt':  # Windows
                    import shutil
                    if item.is_dir():
                        shutil.copytree(item, target)
                    else:
                        shutil.copy2(item, target)
                else:  # Unix-like
                    try:
                        target.symlink_to(item)
                    except OSError:
                        print(f"Could not create symlink. Copying {item.name} instead...")
                        if item.is_dir():
                            import shutil
                            shutil.copytree(item, target)
                        else:
                            import shutil
                            shutil.copy2(item, target)
        
        print(f"Dataset ready at: {data_dir}")

def extract_features(model, data_loader, output_dir, layer_names=None):
    """Extract features from model for given data loader"""
    features_dict = {}
    metadata_dict = {}
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Extracting features"):
            images = batch['image'].to(device)
            metadata = batch['metadata']
            
            # Get features from DeepJuice
            if layer_names:
                activations = model.forward_and_collect(images, layer_names=layer_names)
            else:
                # Get final layer features by default
                activations = model.forward_and_collect(images)
            
            # Store features and metadata
            for i in range(len(metadata['path'])):
                path = metadata['path'][i]
                features_dict[path] = {}
                
                for layer_name, features in activations.items():
                    # Reshape features if needed
                    if len(features.shape) > 2:
                        # For CNNs/ViTs, apply global average pooling
                        feat = features[i].mean(dim=[1, 2]) if len(features.shape) == 4 else features[i].mean(dim=1)
                    else:
                        feat = features[i]
                        
                    features_dict[path][layer_name] = feat.cpu().numpy()
                
                # Store metadata
                metadata_dict[path] = {
                    'subject': metadata['subject'][i],
                    'session': metadata['session'][i],
                    'run': metadata['run'][i],
                    'frame': metadata['frame'][i],
                }
    
    return features_dict, metadata_dict

def save_features(features_dict, metadata_dict, output_dir, model_name):
    """Save extracted features to H5 files"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Group by subject, session, and run
    grouped_features = {}
    for path, features in features_dict.items():
        metadata = metadata_dict[path]
        subject = metadata['subject']
        session = metadata['session']
        run = metadata['run']
        
        key = f"{subject}/{session}/{run}"
        if key not in grouped_features:
            grouped_features[key] = {}
            
        grouped_features[key][metadata['frame']] = features
    
    # Save each group to a separate H5 file
    for key, frames in tqdm(grouped_features.items(), desc="Saving features"):
        output_file = os.path.join(output_dir, f"{model_name}_{key.replace('/', '_')}.h5")
        
        with h5py.File(output_file, 'w') as f:
            for frame, features in frames.items():
                frame_group = f.create_group(frame)
                
                for layer_name, layer_features in features.items():
                    frame_group.create_dataset(layer_name, data=layer_features)
                    
        print(f"Saved features to {output_file}")

def upload_to_huggingface(output_dir, repo_id="medarc/AlgonautsDS-features", token=None):
    """Upload features to HuggingFace datasets"""
    try:
        from huggingface_hub import HfApi, login
        
        if token:
            login(token)
            
        api = HfApi()
        
        # Upload each file in the output directory
        for file_path in tqdm(list(Path(output_dir).glob("*.h5")), desc="Uploading to HuggingFace"):
            api.upload_file(
                path_or_fileobj=str(file_path),
                path_in_repo=str(file_path.name),
                repo_id=repo_id
            )
            
        print(f"Features uploaded to {repo_id}")
    except Exception as e:
        print(f"Error uploading to HuggingFace: {e}")
        print("Please upload the files manually to: https://huggingface.co/datasets/medarc/AlgonautsDS-features/tree/main")

def main():
    # List available models if requested
    if args.list_models:
        print("Available models in DeepJuice model zoo:")
        # Get a subset of vision models for Algonauts
        vision_models = get_model_options().query("source in ['torchvision', 'timm', 'dinov2']")
        print(vision_models[['model_uid', 'source', 'model_name']])
        return
    
    # Download dataset if needed
    if not os.path.exists(args.data_dir):
        download_dataset(args.data_dir, args.use_datalad)
    
    # Set up image preprocessing
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Determine which models to use
    models_to_use = args.models if args.models else [args.model_name]
    
    # Determine which sampling rates to use
    sampling_rates = args.sampling_rates if args.sampling_rates else [args.frame_sampling_rate]
    
    # Process each combination of model and sampling rate
    for model_name in models_to_use:
        for sampling_rate in sampling_rates:
            print(f"\n\n{'='*50}")
            print(f"Processing model: {model_name}, frame sampling rate: {sampling_rate}")
            print(f"{'='*50}\n")
            
            # Create dataset and dataloader with the given sampling rate
            dataset = AlgonautsDataset(args.data_dir, transform=transform, frame_sampling_rate=sampling_rate)
            dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
            
            print(f"Found {len(dataset)} frames in the dataset after sampling")
            
            # Load model using DeepJuice
            try:
                print(f"Loading model: {model_name}")
                # Separately pass the pretrained parameter
                model = get_deepjuice_model(model_name)
                model.to(device)
                model.eval()
                
                # Get all layer names for feature extraction
                layer_names = model.get_layer_names()
                print(f"Model has {len(layer_names)} layers")
                
                # Filter to final layers (can be customized based on needs)
                # For example, get the last convolutional/transformer layers
                final_layers = []
                
                # AlexNet specific layers
                if 'alexnet' in model_name.lower():
                    final_layers = [name for name in layer_names if 'classifier' in name]
                # ResNet specific layers
                elif 'resnet' in model_name.lower():
                    final_layers = [name for name in layer_names if 'layer4' in name or 'fc' in name]
                # ViT specific layers
                elif 'vit' in model_name.lower():
                    final_layers = [name for name in layer_names if 'blocks.11' in name or 'blocks.23' in name or 'heads' in name]
                # Default for other models
                if not final_layers:
                    # Take the last 5 layers if can't find specific ones
                    final_layers = layer_names[-5:]
                
                print(f"Extracting features from layers: {final_layers}")
                
                # Create model-specific output directory
                model_output_dir = os.path.join(args.output_dir, f"{model_name}_sr{sampling_rate}")
                os.makedirs(model_output_dir, exist_ok=True)
                
                # Extract features
                features_dict, metadata_dict = extract_features(model, dataloader, model_output_dir, final_layers)
                
                # Save features
                save_features(features_dict, metadata_dict, model_output_dir, model_name)
                
                # Upload to HuggingFace if requested
                if args.upload_hf:
                    upload_to_huggingface(model_output_dir, token=args.hf_token)
                    
            except Exception as e:
                print(f"Error processing model {model_name}: {e}")
                import traceback
                traceback.print_exc()
                continue

if __name__ == "__main__":
    main() 
