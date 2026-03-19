import argparse
import os

import torch
import numpy as np

import dataset_loader
import model
import util

def eval_9d():
    parser = argparse.ArgumentParser(description="Evaluate DirectionNet PyTorch")
    parser.add_argument('--eval_data_dir', type=str, default='data/R90_fov90/test', help='The eval data directory.')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory to load weights from.')
    parser.add_argument('--batch', type=int, default=1, help='Mini-batch size.')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device {device}")
    
    # Init Data Loader
    loader = dataset_loader.data_loader(args.eval_data_dir, epochs=1, batch_size=args.batch, training=False)
    
    # Init Model (3 output distributions for 9D rotation)
    net = model.DirectionNet(n_out=3).to(device)
    
    # Attempt to load latest checkpoint if exists
    if os.path.exists(args.checkpoint_dir):
        checkpoints = sorted([f for f in os.listdir(args.checkpoint_dir) if f.endswith('.pth')])
        if checkpoints:
            latest_ckpt = os.path.join(args.checkpoint_dir, checkpoints[-1])
            print(f"Loading {latest_ckpt}")
            net.load_state_dict(torch.load(latest_ckpt, map_location=device))
    net.eval()
    
    print('Starting evaluation...')
    
    rot_errors = []
    
    with torch.no_grad():
        for step, batch_data in enumerate(loader):
            src_img = batch_data.src_image.to(device)
            trt_img = batch_data.trt_image.to(device)
            rotation_gt = batch_data.rotation.to(device) # [B, 3, 3]
            
            # Forward
            pred = net(src_img, trt_img) # [B, 3, H, W]
            
            directions, expectation, distribution_pred = util.distributions_to_directions(pred)
            
            # Metrics
            rotation_estimated = util.svd_orthogonalize(directions)
            
            rot_err = util.rotation_geodesic(rotation_estimated, rotation_gt)
            rot_errors.append(util.radians_to_degrees(rot_err).cpu().numpy())
            
    if len(rot_errors) > 0:
        rot_errors = np.concatenate(rot_errors)
        print(f"Evaluation Complete!")
        print(f"Mean Rotation Error (deg): {rot_errors.mean():.4f}")
        print(f"Median Rotation Error (deg): {np.median(rot_errors):.4f}")
    else:
        print("No evaluation data found.")

if __name__ == '__main__':
    eval_9d()
