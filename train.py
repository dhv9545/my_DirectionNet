import argparse
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim

import dataset_loader
import losses
import model
import util

def train_9d():
    parser = argparse.ArgumentParser(description="Train DirectionNet PyTorch")
    parser.add_argument('--data_dir', type=str, default='data/R90_fov90/test', help='The training data directory.')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory to save weights')
    parser.add_argument('--batch', type=int, default=2, help='Mini-batch size.')
    parser.add_argument('--n_epoch', type=int, default=10, help='Number of epochs.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate.')
    parser.add_argument('--alpha', type=float, default=8e7, help='Distribution loss weight.')
    parser.add_argument('--beta', type=float, default=0.1, help='Spread loss weight.')
    parser.add_argument('--kappa', type=float, default=10.0, help='VMF concentration.')
    parser.add_argument('--dist_h', type=int, default=64, help='Output dist height.')
    parser.add_argument('--dist_w', type=int, default=64, help='Output dist width.')
    
    args = parser.parse_args()
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device {device}")
    
    # Init Data Loader
    loader = dataset_loader.data_loader(args.data_dir, epochs=args.n_epoch, batch_size=args.batch, training=True)
    
    # Init Model (3 output distributions for 9D rotation)
    net = model.DirectionNet(n_out=3).to(device)
    optimizer = optim.SGD(net.parameters(), lr=args.lr) # Standard Gradient Descent
    
    print('Starting training loop...')
    global_step = 0
    
    for epoch in range(args.n_epoch):
        for step, batch_data in enumerate(loader):
            start_t = time.time()
            
            src_img = batch_data.src_image.to(device)
            trt_img = batch_data.trt_image.to(device)
            rotation_gt = batch_data.rotation.to(device) # [B, 3, 3]
            
            directions_gt = rotation_gt # For 9D we use full 3x3 as 3 direction vectors
            # GT Distribution
            vmf_prob = util.von_mises_fisher(directions_gt, args.kappa, [args.dist_h, args.dist_w])
            distribution_gt = util.spherical_normalization(vmf_prob, rectify=False)
            
            optimizer.zero_grad()
            
            # Forward
            pred = net(src_img, trt_img) # [B, 3, H, W]
            
            directions, expectation, distribution_pred = util.distributions_to_directions(pred)
            
            # Loss computation
            dir_loss = losses.direction_loss(directions, directions_gt)
            dist_loss = args.alpha * losses.distribution_loss(distribution_pred, distribution_gt)
            spread_loss = args.beta * losses.spread_loss(expectation)
            
            loss = dir_loss + dist_loss + spread_loss
            
            # Metrics
            rotation_estimated = util.svd_orthogonalize(directions)
            rot_err = util.rotation_geodesic(rotation_estimated, rotation_gt).mean()
            dir_err = torch.acos(torch.clamp(torch.sum(directions * directions_gt, dim=-1), -1., 1.)).mean()
            
            # Backward
            loss.backward()
            optimizer.step()
            
            d_time = time.time() - start_t
            
            if global_step % 10 == 0:
                print(f"Epoch {epoch} Step {global_step} | Loss: {loss.item():.4f} "
                      f"| Rot Err (deg): {util.radians_to_degrees(rot_err).item():.2f} "
                      f"| Time: {d_time:.3f}s")
            
            global_step += 1
            
        # Save epoch checkpoint
        ckpt_path = os.path.join(args.checkpoint_dir, f'model_epoch_{epoch}.pth')
        torch.save(net.state_dict(), ckpt_path)
        print(f"Saved {ckpt_path}")

if __name__ == '__main__':
    train_9d()
