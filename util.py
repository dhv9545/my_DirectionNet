# coding=utf-8
import math
import pickle
import torch
import torch.nn.functional as F
import numpy as np

def read_pickle(file):
    with open(file, 'rb') as f:
        loaded = pickle.load(f, encoding='bytes') 
    return list(loaded.keys()), list(loaded.values())

def safe_sqrt(x):
    return torch.sqrt(torch.maximum(x, torch.tensor(1e-20, device=x.device, dtype=x.dtype)))

def degrees_to_radians(degree):
    return math.pi * degree / 180.0

def radians_to_degrees(radians):
    return 180.0 * radians / math.pi

def angular_distance(v1, v2):
    dot = torch.sum(v1 * v2, dim=-1)
    return torch.acos(torch.clamp(dot, -1., 1.))

def equirectangular_area_weights(height, device=None, dtype=torch.float32):
    pixel_h = math.pi / float(height)
    colatitude = torch.linspace(pixel_h / 2, math.pi - pixel_h / 2, height, device=device, dtype=dtype)
    # Return shape compatible with [BATCH, CHANNELS, HEIGHT, WIDTH] -> [1, 1, HEIGHT, 1]
    return torch.sin(colatitude).view(1, 1, height, 1)

def spherical_normalization(x, rectify=True):
    # x is [B, C, H, W]
    if rectify:
        x = F.softplus(x)
        
    _, _, height, _ = x.shape
    weights = equirectangular_area_weights(height, device=x.device, dtype=x.dtype)
    weighted = x * weights
    
    sum_weighted = torch.sum(weighted, dim=[2, 3], keepdim=True)
    # prevent div by zero
    sum_weighted = torch.where(sum_weighted == 0, torch.ones_like(sum_weighted), sum_weighted)
    return x / sum_weighted

def generate_equirectangular_grid(shape, device=None, dtype=torch.float32):
    # shape: [HEIGHT, WIDTH]
    height, width = shape
    pixel_w = 2 * math.pi / float(width)
    pixel_h = math.pi / float(height)
    
    colatitude = torch.linspace(pixel_h / 2, math.pi - pixel_h / 2, height, device=device, dtype=dtype)
    azimuth = torch.linspace(pixel_w / 2, 2 * math.pi - pixel_w / 2, width, device=device, dtype=dtype)
    
    # meshgrid indexing 'ij' gives [H, W] mapping for both.
    colat_grid, azi_grid = torch.meshgrid(colatitude, azimuth, indexing='ij')
    
    return torch.stack([colat_grid, azi_grid], dim=-1) # [H, W, 2]

def spherical_to_cartesian(spherical):
    # spherical: [..., 2]
    colatitude = spherical[..., 0]
    azimuth = spherical[..., 1]
    
    x = torch.sin(colatitude) * torch.cos(azimuth)
    # Note: original TFG does:
    y = torch.sin(colatitude) * torch.sin(azimuth)
    z = torch.cos(colatitude)
    return torch.stack([x, y, z], dim=-1)

def spherical_expectation(spherical_probabilities):
    # spherical_probabilities: [BATCH, CHANNELS, HEIGHT, WIDTH]
    batch, channels, height, width = spherical_probabilities.shape
    
    spherical = generate_equirectangular_grid([height, width], device=spherical_probabilities.device, dtype=spherical_probabilities.dtype)
    unit_directions = spherical_to_cartesian(spherical) # [H, W, 3]
    
    # Original axis convert: [[1,0,0],[0,0,-1],[0,1,0]] transposed
    # So multiply cartesian by this transposed matrix
    axis_convert = torch.tensor([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]], device=spherical_probabilities.device, dtype=spherical_probabilities.dtype)
    
    # unit_directions [H, W, 3] -> multiply by axis_convert.T to match TF behavior
    # TF: matmul(axis_convert, expand_dims(cartesian, -1), transpose_a=True) 
    # = axis_convert^T * cartesian
    unit_directions = torch.matmul(unit_directions, axis_convert) # Because X * A is X^T * A ... wait.
    # torch.matmul(unit_directions, axis_convert) = sum(unit_directions[i] * axis_convert[i, j])
    # TFG mathematically: axis_convert^T * cartesian (column vec). In PyTorch dotting on last dimension of [H,W,3] with a [3,3] matrix:
    # (H,W,3) @ (3,3) -> multiplies each 3-vec (row) by 3x3 matrix.
    # To precisely match `axis_convert^T x`, we do `unit_directions @ axis_convert`
    
    unit_directions = unit_directions.view(1, 1, height, width, 3)
    
    weights = equirectangular_area_weights(height, device=spherical_probabilities.device, dtype=spherical_probabilities.dtype)
    weighted = spherical_probabilities * weights # [BATCH, CHANNELS, HEIGHT, WIDTH]
    
    weighted = weighted.unsqueeze(-1) # [BATCH, CHANNELS, HEIGHT, WIDTH, 1]
    
    expectation = torch.sum(unit_directions * weighted, dim=[2, 3]) # [BATCH, CHANNELS, 3]
    return expectation

def von_mises_fisher(mean, concentration, shape):
    # mean: [BATCH, CHANNELS, 3]
    # shape: [HEIGHT, WIDTH]
    # returns: [BATCH, CHANNELS, HEIGHT, WIDTH]
    batch, channels = mean.shape[0], mean.shape[1]
    height, width = shape
    
    spherical_grid = generate_equirectangular_grid([height, width], device=mean.device, dtype=mean.dtype)
    cartesian = spherical_to_cartesian(spherical_grid)
    
    axis_convert = torch.tensor([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]], device=mean.device, dtype=mean.dtype)
    cartesian = torch.matmul(cartesian, axis_convert) # [H, W, 3]
    
    cartesian = cartesian.view(1, 1, height, width, 3) 
    mean = mean.view(batch, channels, 1, 1, 3)
    
    dot_product = torch.sum(mean * cartesian, dim=-1) # [BATCH, CHANNELS, HEIGHT, WIDTH]
    
    C = concentration / (4 * math.pi * math.sinh(concentration))
    prob = C * torch.exp(concentration * dot_product)
    
    return prob

def rotation_geodesic(r1, r2):
    # r1, r2: [BATCH, 3, 3]
    # trace of (r1 @ r2^T)
    # output: [BATCH]
    r2_t = r2.transpose(1, 2)
    m = torch.bmm(r1, r2_t)
    trace = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]
    diff = (trace - 1) / 2
    return torch.acos(torch.clamp(diff, -1.0 + 1e-7, 1.0 - 1e-7))

def gram_schmidt(m):
    # m: [BATCH, 2, 3]
    x = m[:, 0]
    y = m[:, 1]
    xn = F.normalize(x, dim=-1)
    z = torch.linalg.cross(xn, y)
    zn = F.normalize(z, dim=-1)
    y_new = torch.linalg.cross(zn, xn)
    return torch.stack([xn, y_new, zn], dim=1)

def svd_orthogonalize(m):
    # m: [BATCH, 3, 3]
    m_norm = F.normalize(m, dim=-1)
    m_transpose = m_norm.transpose(-1, -2)
    U, S, Vh = torch.linalg.svd(m_transpose)
    
    V = Vh.transpose(-1, -2)
    
    det = torch.linalg.det(torch.bmm(V, U.transpose(-1, -2)))
    
    # Check orientation reflection
    v_modified = torch.cat([V[:, :, :-1], V[:, :, -1:] * det.unsqueeze(-1).unsqueeze(-1)], dim=2)
    r = torch.bmm(v_modified, U.transpose(-1, -2))
    return r

def distributions_to_directions(x):
    distribution_pred = spherical_normalization(x)
    expectation = spherical_expectation(distribution_pred)
    expectation_normalized = F.normalize(expectation, dim=-1)
    return expectation_normalized, expectation, distribution_pred


# --- Transformations for Translation Model ---
def generate_cartesian_grid_pt(resolution, fov, device='cpu'):
    H, W = resolution
    fov_rad = fov * math.pi / 180.0
    width = 2 * math.tan(fov_rad / 2.0)
    height = width * H / W
    px_w = width / W
    px_h = height / H
    x = torch.linspace(-width/2 + px_w/2, width/2 - px_w/2, W, device=device)
    y = torch.linspace(height/2 - px_h/2, -height/2 + px_h/2, H, device=device)
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    grid = torch.stack([xx, yy, -torch.ones_like(xx)], dim=-1)
    return grid

def rotate_image_in_3d_pt(images, input_rotations, input_fov, output_fov, output_shape):
    B, C, in_H, in_W = images.shape
    device = images.device
    out_H, out_W = output_shape
    cartesian = generate_cartesian_grid_pt(output_shape, output_fov, device)
    cartesian = cartesian.view(1, out_H, out_W, 3).expand(B, -1, -1, -1)
    cartesian_flat = cartesian.reshape(B, -1, 3).transpose(1, 2)
    rotated_coords = torch.bmm(input_rotations.transpose(1, 2), cartesian_flat)
    rotated_coords = rotated_coords.transpose(1, 2).reshape(B, out_H, out_W, 3)
    x = -rotated_coords[..., 0] / rotated_coords[..., 2]
    y = -rotated_coords[..., 1] / rotated_coords[..., 2]
    in_fov_rad = input_fov * math.pi / 180.0
    in_w = 2 * torch.tan(in_fov_rad / 2.0).view(B, 1, 1)
    in_h = 2 * torch.tan(in_fov_rad / 2.0).view(B, 1, 1)
    grid_x = x / (in_w / 2.0)
    grid_y = y / (in_h / 2.0)
    grid_y = -grid_y
    grid = torch.stack([grid_x, grid_y], dim=-1)
    return torch.nn.functional.grid_sample(images, grid, mode='bilinear', padding_mode='zeros', align_corners=False)

def half_rotation_pt(rot_matrix):
    B = rot_matrix.shape[0]
    dev = rot_matrix.device
    trace = rot_matrix.diagonal(dim1=-2, dim2=-1).sum(-1)
    theta = torch.acos(torch.clamp((trace - 1.0) / 2.0, -1.0+1e-6, 1.0-1e-6))
    R_skew = rot_matrix - rot_matrix.transpose(1, 2)
    sin_theta = torch.sin(theta).unsqueeze(-1)
    axis = torch.stack([R_skew[:, 2, 1], R_skew[:, 0, 2], R_skew[:, 1, 0]], dim=-1)
    axis = axis / (2 * sin_theta + 1e-8)
    half_theta = theta / 2.0
    K = torch.zeros(B, 3, 3, device=dev)
    K[:, 0, 1] = -axis[:, 2]
    K[:, 0, 2] = axis[:, 1]
    K[:, 1, 0] = axis[:, 2]
    K[:, 1, 2] = -axis[:, 0]
    K[:, 2, 0] = -axis[:, 1]
    K[:, 2, 1] = axis[:, 0]
    I = torch.eye(3, device=dev).unsqueeze(0).expand(B, -1, -1)
    sin_h = torch.sin(half_theta).view(B, 1, 1)
    cos_h = torch.cos(half_theta).view(B, 1, 1)
    half_R = I + sin_h * K + (1.0 - cos_h) * torch.bmm(K, K)
    mask = (theta < 1e-5).view(B, 1, 1)
    return torch.where(mask, I, half_R)

def derotation(src_img, trt_img, rotation, input_fov, output_fov, output_shape, derotate_both):
    B = src_img.shape[0]
    device = src_img.device
    if derotate_both:
        half_derotation = half_rotation_pt(rotation)
        transformed_src = rotate_image_in_3d_pt(src_img, half_derotation.transpose(1, 2), input_fov, output_fov, output_shape)
        transformed_trt = rotate_image_in_3d_pt(trt_img, half_derotation, input_fov, output_fov, output_shape)
    else:
        eye = torch.eye(3, device=device).unsqueeze(0).expand(B, -1, -1)
        transformed_src = rotate_image_in_3d_pt(src_img, eye, input_fov, output_fov, output_shape)
        transformed_trt = rotate_image_in_3d_pt(trt_img, rotation, input_fov, output_fov, output_shape)
    return transformed_src, transformed_trt
