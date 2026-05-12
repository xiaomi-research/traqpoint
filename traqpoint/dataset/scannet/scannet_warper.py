import torch
from kornia.utils import create_meshgrid
import matplotlib.pyplot as plt
import pdb
from .utils import warp


import os
import torchvision.transforms as transforms
from matplotlib.patches import ConnectionPatch

@torch.no_grad()
def spvs_coarse(data, scale = 8):
    N, _, H0, W0 = data['image0'].shape
    _, _, H1, W1 = data['image1'].shape
    device = data['image0'].device
    corrs = []
    for idx in range(N):
        warp01_params = {}
        for k, v in data['warp01_params'].items():
            if isinstance(v[idx], torch.Tensor):
                warp01_params[k] = v[idx].to(device)
            else:
                warp01_params[k] = v[idx]
        warp10_params = {}
        for k, v in data['warp10_params'].items():
            if isinstance(v[idx], torch.Tensor):
                warp10_params[k] = v[idx].to(device)
            else:
                warp10_params[k] = v[idx]
            
        # create kpts
        h0, w0, h1, w1 = map(lambda x: x // scale, [H0, W0, H1, W1])
        grid_pt1_c = create_meshgrid(h1, w1, False, device).reshape(h1*w1, 2)    # [N, hw, 2]
        
        # normalize kpts
        grid_pt1_c = grid_pt1_c * scale

        # try:
        if 1:
            grid_pt1_c_valid, grid_pt10_c, ids1, ids1_out = warp(grid_pt1_c, warp10_params)
            grid_pt10_c_valid, grid_pt01_c, ids0, ids0_out = warp(grid_pt10_c, warp01_params)
            
            # check reproj error
            grid_pt1_c_valid = grid_pt1_c_valid[ids0]
            dist = torch.linalg.norm(grid_pt1_c_valid - grid_pt01_c, dim=-1)
            
            mask_mutual = (dist < 1.5) 
            
            #get correspondences
            pts = torch.cat([grid_pt10_c_valid[mask_mutual] / scale,
                                grid_pt01_c[mask_mutual] / scale], dim=-1)
            #remove repeated correspondences
            lut_mat12 = torch.ones((h1, w1, 4), device = device, dtype = torch.float32) * -1
            lut_mat21 = torch.clone(lut_mat12)
            src_pts = pts[:, :2]
            tgt_pts = pts[:, 2:]
        
            lut_mat12[src_pts[:,1].long(), src_pts[:,0].long()] = torch.cat([src_pts, tgt_pts], dim=1)
            mask_valid12 = torch.all(lut_mat12 >= 0, dim=-1)
            points = lut_mat12[mask_valid12]

            #Target-src check
            src_pts, tgt_pts = points[:, :2], points[:, 2:]
            lut_mat21[tgt_pts[:,1].long(), tgt_pts[:,0].long()] = torch.cat([src_pts, tgt_pts], dim=1)
            mask_valid21 = torch.all(lut_mat21 >= 0, dim=-1)
            points = lut_mat21[mask_valid21]
            corrs.append(points)

            # Extract current sample's images and points
            img0 = data['image0'][idx]
            img1 = data['image1'][idx]
            pts0 = points[:, :2] * scale  # Restore to original image scale
            pts1 = points[:, 2:] * scale

            # Generate save path

            save_path = os.path.join("./", f"correspondences_sample_{idx}.png")

            # Save matching point pairs image
            plot_corrs(
                    img0, img1,
                    pts0, pts1,
                    save_path=save_path,
                    num_points=300,
                    title=f"Correspondences for sample {idx}"
            )
            print(f"Matching point pairs image saved to: {save_path}")
            import pdb;pdb.set_trace()
        # except:
        #     corrs.append(torch.zeros((0, 4), device = device))
          

    #Plot for debug purposes    
    # for i in range(len(corrs)):
    #     plot_corrs(data['image0'][i], data['image1'][i], corrs[i][:, :2]*8, corrs[i][:, 2:]*8)

    return corrs



def plot_corrs(img0, img1, pts0, pts1, save_path, num_points=50, title="Correspondences"):
    """
    Draw corresponding point pairs on two images and connect them with lines, save to file

    Args:
        img0: First image (tensor)
        img1: Second image (tensor)
        pts0: Points on first image (tensor, shape [N, 2])
        pts1: Points on second image (tensor, shape [N, 2])
        save_path: Image save path
        num_points: Number of point pairs to display, too many will look messy
        title: Chart title
    """
    # Convert tensor to displayable image format
    to_pil = transforms.ToPILImage()
    
    # If images are on GPU, transfer to CPU and convert to numpy
    if img0.device.type == 'cuda':
        img0 = img0.cpu()
        img1 = img1.cpu()
        pts0 = pts0.cpu().numpy()
        pts1 = pts1.cpu().numpy()
    else:
        pts0 = pts0.numpy()
        pts1 = pts1.numpy()
    
    # Convert to PIL images
    img0 = to_pil(img0)
    img1 = to_pil(img1)
    
    # Create canvas
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    ax1.imshow(img0)
    ax2.imshow(img1)
    
    # Limit the number of displayed point pairs to avoid clutter from too many points
    num_points = min(num_points, len(pts0))
    if num_points > 0:  # Ensure there are points to display
        pts0 = pts0[:num_points]
        pts1 = pts1[:num_points]
        
        # Draw points and connect with lines
        for i, (p0, p1) in enumerate(zip(pts0, pts1)):
            # Use different colors for each point pair for easy distinction
            color = plt.cm.hsv(i / num_points)
            
            # Draw point on first image
            ax1.plot(p0[0], p0[1], 'o', markersize=5, color=color, alpha=0.7)
            # Draw point on second image
            ax2.plot(p1[0], p1[1], 'o', markersize=5, color=color, alpha=0.7)
            # Connect with line
            con = ConnectionPatch(xyA=p0, xyB=p1, coordsA="data", coordsB="data",
                                axesA=ax1, axesB=ax2, color=color, linestyle='--', alpha=0.5)
            fig.add_artist(con)
    
    ax1.set_title('Image 0')
    ax2.set_title('Image 1')
    ax1.axis('off')
    ax2.axis('off')
    plt.suptitle(title)
    plt.tight_layout()
    
    # Save image to specified path
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close(fig)  # Close image, release memory