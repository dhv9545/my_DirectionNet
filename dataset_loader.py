import os
import random
import pickle
import glob
from collections import namedtuple

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import torchvision.transforms.functional as F

DataInput = namedtuple(
    'data_input',
    ['id', 'src_image', 'trt_image', 'rotation', 'translation', 'fov', 'rotation_pred']
)

class DirectionNetDataset(Dataset):
    def __init__(self, data_path, training=True, load_estimated_rot=False):
        self.data_path = data_path
        self.training = training
        self.load_estimated_rot = load_estimated_rot
        
        self.samples = []
        
        # Discover all subdirectories
        subfolders = [f.path for f in os.scandir(data_path) if f.is_dir()]
        
        for folder in subfolders:
            try:
                with open(os.path.join(folder, 'rotation_gt.pickle'), 'rb') as f:
                    rotations = pickle.load(f, encoding='bytes')
                with open(os.path.join(folder, 'epipoles_gt.pickle'), 'rb') as f:
                    translations = pickle.load(f, encoding='bytes')
                with open(os.path.join(folder, 'fov.pickle'), 'rb') as f:
                    fovs = pickle.load(f, encoding='bytes')
                    
                rotation_preds = None
                if self.load_estimated_rot:
                    rot_pred_path = os.path.join(folder, 'rotation_pred.pickle')
                    if os.path.exists(rot_pred_path):
                        with open(rot_pred_path, 'rb') as f:
                            rotation_preds = pickle.load(f, encoding='bytes')
                
                for img_id in rotations.keys():
                    str_id = img_id.decode('utf-8') if isinstance(img_id, bytes) else str(img_id)
                    sample = {
                        'id': str_id,
                        'folder': folder,
                        'rotation': rotations[img_id],
                        'translation': translations[img_id],
                        'fov': fovs[img_id]
                    }
                    if self.load_estimated_rot:
                        sample['rotation_pred'] = rotation_preds[img_id] if rotation_preds is not None else torch.zeros(3, 3)
                    else:
                        sample['rotation_pred'] = torch.zeros(3, 3)
                        
                    self.samples.append(sample)
            except Exception as e:
                # If some folder doesn't have the pickles, skip it
                pass

    def __len__(self):
        return len(self.samples)

    def _load_image(self, img_path):
        # returns [3, H, W] tensor in range [0, 1]
        img = read_image(img_path).float() / 255.0
        
        # Resize from [512, 512] to [256, 256]. TF uses resize_area, bilinear is close enough over smaller resolutions.
        img = F.resize(img, [256, 256], antialias=True)
        return img

    def __getitem__(self, idx):
        sample = self.samples[idx]
        folder = sample['folder']
        img_id = sample['id']
        
        src_path = os.path.join(folder, f"{img_id}.src.perspective.png")
        trt_path = os.path.join(folder, f"{img_id}.trt.perspective.png")
        
        src_image = self._load_image(src_path)
        trt_image = self._load_image(trt_path)
        
        if self.training:
            random_gamma = random.uniform(0.7, 1.2)
            src_image = F.adjust_gamma(src_image, random_gamma)
            trt_image = F.adjust_gamma(trt_image, random_gamma)
            
        rot = torch.tensor(sample['rotation'], dtype=torch.float32)
        trans = torch.tensor(sample['translation'], dtype=torch.float32)
        fov = torch.tensor([sample['fov']], dtype=torch.float32)
        rot_pred = torch.tensor(sample['rotation_pred'], dtype=torch.float32)
        
        # TF stack shape adjustments
        rot = rot.view(3, 3)
        trans = trans.view(3)
        fov = fov.view(1)
        rot_pred = rot_pred.view(3, 3)
        
        return DataInput(
            id=img_id,
            src_image=src_image,
            trt_image=trt_image,
            rotation=rot,
            translation=trans,
            fov=fov,
            rotation_pred=rot_pred
        )

def data_loader(data_path, epochs=1, batch_size=32, training=True, load_estimated_rot=False, num_workers=0):
    """
    Load stereo image datasets.
    Returns: PyTorch DataLoader
    """
    dataset = DirectionNetDataset(data_path, training, load_estimated_rot)
    
    # We ignore epochs here because PyTorch typical loops iterate epochs externally.
    # We supply drop_last=True to replicate PyTorch / TF drop_remainder=True.
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=training, 
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True
    )
    return loader
