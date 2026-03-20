import argparse
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
import random

import dataset_loader
import losses
import model
import util

def main():
    parser = argparse.ArgumentParser(description="Train DirectionNet PyTorch")
    parser.add_argument('--data_dir', type=str, default='data/R90_fov90/test', help='The training data directory.')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory to save weights')
    parser.add_argument('--model', type=str, default='9D', choices=['9D', '6D', 'T'], help='Model type: 9D, 6D, or T')
    parser.add_argument('--batch', type=int, default=2, help='Mini-batch size.')
    parser.add_argument('--n_epoch', type=int, default=10, help='Number of epochs.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate.')
    parser.add_argument('--alpha', type=float, default=8e7, help='Distribution loss weight.')
    parser.add_argument('--beta', type=float, default=0.1, help='Spread loss weight.')
    parser.add_argument('--kappa', type=float, default=10.0, help='VMF concentration.')
    parser.add_argument('--dist_h', type=int, default=64, help='Output dist height.')
    parser.add_argument('--dist_w', type=int, default=64, help='Output dist width.')
    
    # Translation model specific args
    parser.add_argument('--transformed_fov', type=float, default=105.0)
    parser.add_argument('--transformed_height', type=int, default=344)
    parser.add_argument('--transformed_width', type=int, default=344)
    parser.add_argument('--derotate_both', action='store_true', default=True)
    parser.add_argument('--no_derotate_both', action='store_false', dest='derotate_both')
    
    args = parser.parse_args()
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device {device}")
    
    # Data Loader needs estimated_rot for 'T'
    load_est_rot = (args.model == 'T')
    loader = dataset_loader.data_loader(args.data_dir, epochs=args.n_epoch, batch_size=args.batch, training=True, load_estimated_rot=load_est_rot)
    
    # Init Model
    if args.model == '9D':
        n_out = 3
    elif args.model == '6D':
        n_out = 2
    elif args.model == 'T':
        n_out = 1
        
    net = model.DirectionNet(n_out=n_out).to(device)
    optimizer = optim.SGD(net.parameters(), lr=args.lr)
    
    print(f'Starting training loop for {args.model}...')
    global_step = 0
    
    # We will perturb rotations 50% of the time during Translation training
    def perturb_rot(rotations):
        # Rough translation of util.perturb_rotation logic: Add some noise.
        # For simplicity in this rewrite, we will just use the estimated rotation normally
        return rotations

    for epoch in range(args.n_epoch):
        for step, batch_data in enumerate(loader):
            start_t = time.time()
            
            src_img = batch_data.src_image.to(device)
            trt_img = batch_data.trt_image.to(device)
            rotation_gt = batch_data.rotation.to(device) # [B, 3, 3]
            
            if args.model in ['9D', '6D']:
                directions_gt = rotation_gt[:, :n_out, :]
            elif args.model == 'T':
                translation_gt = batch_data.translation.to(device)
                directions_gt = translation_gt.unsqueeze(1) # [B, 1, 3]
                fov_gt = batch_data.fov.to(device)
                rotation_pred = batch_data.rotation_pred.to(device)
                
                # Derotation
                perturbed_rotation = rotation_gt if random.random() < 0.5 else rotation_pred
                src_img, trt_img = util.derotation(
                    src_img, trt_img, perturbed_rotation, fov_gt,
                    args.transformed_fov, [args.transformed_height, args.transformed_width],
                    args.derotate_both
                )
            
            # GT Distribution
            vmf_prob = util.von_mises_fisher(directions_gt, args.kappa, [args.dist_h, args.dist_w])
            distribution_gt = util.spherical_normalization(vmf_prob, rectify=False)
            
            optimizer.zero_grad()
            
            # Forward
            pred = net(src_img, trt_img) # [B, n_out, H, W]
            
            directions, expectation, distribution_pred = util.distributions_to_directions(pred)
            
            # Loss computation
            dir_loss = losses.direction_loss(directions, directions_gt)
            dist_loss = args.alpha * losses.distribution_loss(distribution_pred, distribution_gt)
            spread_loss = args.beta * losses.spread_loss(expectation)
            
            loss = dir_loss + dist_loss + spread_loss
            
            # Metrics
            if args.model == '9D':
                rotation_estimated = util.svd_orthogonalize(directions)
                err = util.rotation_geodesic(rotation_estimated, rotation_gt).mean()
                metric_name = "Rot Err (deg)"
            elif args.model == '6D':
                rotation_estimated = util.gram_schmidt(directions)
                err = util.rotation_geodesic(rotation_estimated, rotation_gt).mean()
                metric_name = "Rot Err (deg)"
            elif args.model == 'T':
                dir_err = torch.acos(torch.clamp(torch.sum(directions * directions_gt, dim=-1), -1., 1.)).mean()
                err = dir_err
                metric_name = "Trans Err (deg)"
            
            # Backward
            loss.backward()
            optimizer.step()
            
            d_time = time.time() - start_t
            
            if global_step % 10 == 0:
                print(f"Epoch {epoch} Step {global_step} | Loss: {loss.item():.4f} "
                      f"| {metric_name}: {util.radians_to_degrees(err).item():.2f} "
                      f"| Time: {d_time:.3f}s")
            
            global_step += 1
            
        # Save epoch checkpoint
        ckpt_path = os.path.join(args.checkpoint_dir, f'model_{args.model}_epoch_{epoch}.pth')
        torch.save(net.state_dict(), ckpt_path)
        print(f"Saved {ckpt_path}")

if __name__ == '__main__':
    main()
