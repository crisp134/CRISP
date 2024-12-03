from .net import CyCNN, MLP, UNet
import torch
import torch.nn as nn
from torchvision import transforms as T

class Renderer(nn.Module):
    """
    Renderer class implementing parallel Cy-CNNs for x, y, z axes,
    followed by MLP and U-Net for radiance mapping and refinement.
    """

    def __init__(self, args):
        super(Renderer, self).__init__()
        # Initialize CyCNN networks for x, y, and z cylindrical projections
        self.cyc_x = CyCNN(input_channels=7, output_channels=args.dim).to(args.device)
        self.cyc_y = CyCNN(input_channels=7, output_channels=args.dim).to(args.device)
        self.cyc_z = CyCNN(input_channels=7, output_channels=args.dim).to(args.device)

        # Initialize MLP and U-Net networks
        self.mlp = MLP(args.dim).to(args.device)
        self.unet = UNet(args).to(args.device)

        # Setup parameters and utilities for image transformation
        self.dim = args.dim
        self.pad_w = T.Pad(args.pad, 1.0, 'constant')
        self.pad_b = T.Pad(args.pad, -1.0, 'constant')

        # Training configurations
        self.train_size = args.train_size
        if args.xyznear:
            self.randomcrop = T.RandomResizedCrop(
                args.train_size, scale=(args.scale_min, args.scale_max), ratio=(1.0, 1.0)
            )
        else:
            self.randomcrop = T.RandomResizedCrop(
                args.train_size,
                scale=(args.scale_min, args.scale_max),
                ratio=(1.0, 1.0),
                interpolation=T.InterpolationMode.NEAREST,
            )

    def perform_cylindrical_projection(self, point_cloud):
        """
        Perform cylindrical projection for the given point cloud along x, y, z axes.
        Args:
            point_cloud: Original 3D point cloud data
        Returns:
            A list containing cylindrical projections for x, y, and z axes.
        """
        # Cylindrical projection along the x-axis
        theta_x = torch.atan2(point_cloud[..., 2], point_cloud[..., 1])
        h_x = point_cloud[..., 0]
        rho_x = torch.ones_like(h_x) * 1.0  # Assume fixed radius
        cyl_proj_x = torch.stack([rho_x, theta_x, h_x, point_cloud[..., 3:]], dim=-1)

        # Cylindrical projection along the y-axis
        theta_y = torch.atan2(point_cloud[..., 0], point_cloud[..., 2])
        h_y = point_cloud[..., 1]
        rho_y = torch.ones_like(h_y) * 1.0  # Assume fixed radius
        cyl_proj_y = torch.stack([rho_y, theta_y, h_y, point_cloud[..., 3:]], dim=-1)

        # Cylindrical projection along the z-axis
        theta_z = torch.atan2(point_cloud[..., 1], point_cloud[..., 0])
        h_z = point_cloud[..., 2]
        rho_z = torch.ones_like(h_z) * 1.0  # Assume fixed radius
        cyl_proj_z = torch.stack([rho_z, theta_z, h_z, point_cloud[..., 3:]], dim=-1)

        return [cyl_proj_x, cyl_proj_y, cyl_proj_z]

    def forward(self, point_cloud, cylindrical_images, ray, mask_gt=None, isTrain=True):
        """
        Args:
            point_cloud: Original 3D point cloud data
            cylindrical_images: List of cylindrical projections for x, y, and z axes
            ray: Ray direction map
            mask_gt: Ground truth mask
            isTrain: Whether in training mode

        Returns:
            A dictionary containing:
                - img: Rendered image
                - gt: Ground truth image after cropping and resizing
                - mask_gt: Ground truth mask after cropping and resizing
                - fea_map: First three dimensions of the feature map after radiance mapping
        """
        # Step 1: Perform Cylindrical Projections
        cylindrical_images = self.perform_cylindrical_projection(point_cloud)
        
        # Step 2: Cy-CNN Processing
        feature_x = self.cyc_x(cylindrical_images[0])  # Process x-axis cylindrical projection
        feature_y = self.cyc_y(cylindrical_images[1])  # Process y-axis cylindrical projection
        feature_z = self.cyc_z(cylindrical_images[2])  # Process z-axis cylindrical projection

        # Combine the features from all three axes
        combined_features = torch.max(torch.stack([feature_x, feature_y, feature_z], dim=0), dim=0)[0]

        # Step 3: MLP for Feature Extraction
        radiance_features = self.mlp(point_cloud, combined_features)

        # Step 4: U-Net for Image Refinement
        refined_features = self.unet(radiance_features.permute(2, 3, 0, 1))  # Permute for U-Net input format

        if isTrain:
            # Padding and Cropping for Training Mode
            dirs = self.pad_w(ray[..., 3:6].permute(2, 0, 1).unsqueeze(0))
            cos = self.pad_w(ray[..., -1:].permute(2, 0, 1).unsqueeze(0))
            gt = self.pad_w(cylindrical_images[0].permute(2, 0, 1).unsqueeze(0))
            zbuf = self.pad_b(cylindrical_images[1].permute(2, 0, 1).unsqueeze(0))

            if mask_gt is not None:
                mask_gt = mask_gt.permute(2, 0, 1).unsqueeze(0)
                cat_img = torch.cat([dirs, cos, gt, zbuf, mask_gt], dim=1)
            else:
                cat_img = torch.cat([dirs, cos, gt, zbuf], dim=1)

            cat_img = self.randomcrop(cat_img)

            # Update variables after cropping
            _, _, H, W = cat_img.shape
            dirs = cat_img[0, :3].permute(1, 2, 0)
            cos = cat_img[0, 3:4].permute(1, 2, 0)
            gt = cat_img[0, 4:7].permute(1, 2, 0)
            zbuf = cat_img[0, 7:8].permute(1, 2, 0)
            if mask_gt is not None:
                mask_gt = cat_img[0, 8:].permute(1, 2, 0)
            pix_mask = zbuf > 0.2

        else:
            # Extract features without cropping for testing/inference
            H, W, K = cylindrical_images[0].shape
            dirs = ray[..., 3:6]
            cos = ray[..., -1:]
            pix_mask = zbuf > 0

        # Generate the final output image from refined features
        img = refined_features.squeeze(0).permute(1, 2, 0)

        return {'img': img, 'gt': gt, 'mask_gt': mask_gt, 'fea_map': radiance_features[..., :3]}
