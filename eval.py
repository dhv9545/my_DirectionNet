import argparse
import os

import torch
import numpy as np

import dataset_loader
import model
import util

def main():
    parser = argparse.ArgumentParser(description="Evaluate DirectionNet PyTorch")
    parser.add_argument('--eval_data_dir', type=str, default='data/R90_fov90/test', help='The eval data directory.')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory to load weights from.')
    parser.add_argument('--model', type=str, default='9D', choices=['9D', '6D', 'T'], help='Model type: 9D, 6D, or T')
    parser.add_argument('--batch', type=int, default=1, help='Mini-batch size.')
    
    # Translation model specific args
    parser.add_argument('--transformed_fov', type=float, default=105.0)
    parser.add_argument('--transformed_height', type=int, default=344)
    parser.add_argument('--transformed_width', type=int, default=344)
    parser.add_argument('--derotate_both', action='store_true', default=True)
    parser.add_argument('--no_derotate_both', action='store_false', dest='derotate_both')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device {device}")
    
    load_est_rot = (args.model == 'T')
    loader = dataset_loader.data_loader(args.eval_data_dir, epochs=1, batch_size=args.batch, training=False, load_estimated_rot=load_est_rot)
    
    if args.model == '9D':
        n_out = 3
    elif args.model == '6D':
        n_out = 2
    elif args.model == 'T':
        n_out = 1
        
    net = model.DirectionNet(n_out=n_out).to(device)
    
    if os.path.exists(args.checkpoint_dir):
        # Prefer loading the checkpoint specific to this model
        checkpoints = sorted([f for f in os.listdir(args.checkpoint_dir) if f.endswith('.pth') and f'model_{args.model}' in f])
        if not checkpoints:
            checkpoints = sorted([f for f in os.listdir(args.checkpoint_dir) if f.endswith('.pth')])
            
        if checkpoints:
            latest_ckpt = os.path.join(args.checkpoint_dir, checkpoints[-1])
            print(f"Loading {latest_ckpt}")
            net.load_state_dict(torch.load(latest_ckpt, map_location=device))
    net.eval()
    
    print(f'Starting evaluation for {args.model}...')
    
    errors = []
    
    with torch.no_grad():
        for step, batch_data in enumerate(loader):
            src_img = batch_data.src_image.to(device)
            trt_img = batch_data.trt_image.to(device)
            
            if args.model == 'T':
                translation_gt = batch_data.translation.to(device)
                directions_gt = translation_gt.unsqueeze(1)
                rotation_gt = batch_data.rotation.to(device)
                fov_gt = batch_data.fov.to(device)
                rotation_pred = batch_data.rotation_pred.to(device)
                
                src_img, trt_img = util.derotation(
                    src_img, trt_img, rotation_pred, fov_gt,
                    args.transformed_fov, [args.transformed_height, args.transformed_width],
                    args.derotate_both
                )
            elif args.model in ['9D', '6D']:
                rotation_gt = batch_data.rotation.to(device)
            
            # Forward
            pred = net(src_img, trt_img) # [B, n_out, H, W]
            
            directions, expectation, distribution_pred = util.distributions_to_directions(pred)
            
            # Metrics
            if args.model == '9D':
                rotation_estimated = util.svd_orthogonalize(directions)
                err = util.rotation_geodesic(rotation_estimated, rotation_gt)
            elif args.model == '6D':
                rotation_estimated = util.gram_schmidt(directions)
                err = util.rotation_geodesic(rotation_estimated, rotation_gt)
            elif args.model == 'T':
                err = torch.acos(torch.clamp(torch.sum(directions * directions_gt, dim=-1), -1., 1.))
                
            errors.append(util.radians_to_degrees(err).cpu().numpy())
            
    if len(errors) > 0:
        errors = np.concatenate(errors)
        print(f"Evaluation Complete!")
        print(f"Mean Error (deg): {errors.mean():.4f}")
        print(f"Median Error (deg): {np.median(errors):.4f}")
    else:
        print("No evaluation data found.")

if __name__ == '__main__':
    main()
