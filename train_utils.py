import torch
import torch.nn as nn
import torch.distributed as dist
from einops import rearrange
from timm.models.layers import to_2tuple


class AllGather_multi(torch.autograd.Function):
    """An autograd function that performs allgather on a tensor."""

    @staticmethod
    def forward(ctx, tensor, args):
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        output = [torch.empty_like(tensor) for _ in range(args.world_size)]
        dist.all_gather(output, tensor)
        ctx.rank = args.rank
        ctx.batch_size = tensor.shape[0]
        return torch.cat(output, 0)

    @staticmethod
    def backward(ctx, grad_output):
        return (
            grad_output[ctx.batch_size * ctx.rank : ctx.batch_size * (ctx.rank + 1)],
            None, None,
        )


def sim_matrix(a, b, eps=1e-8):
        """
        added eps for numerical stability
        """
        a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
        a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
        b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
        sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
        return sim_mt


class PatchifyObjectMask(nn.Module):
    """ Video to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=1, num_frames=8, obj_threshold=0.05, device='cuda'):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0]) * num_frames
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.num_frames = num_frames
        self.in_chans = in_chans
        self.obj_threshold = obj_threshold
        self.filter = torch.ones((in_chans, 1, patch_size[0], patch_size[1])).to(device)

    def forward(self, x):
        B, F, O, H, W = x.shape
        assert F <= self.num_frames
        x = rearrange(x, 'b f o h w -> (b f o) 1 h w')
        # Convert x from boolean to float
        x = x.float()
        x = nn.functional.conv2d(x, self.filter, stride=self.patch_size[0])
        x = x > self.obj_threshold
        x = rearrange(x, '(b t o) 1 h w -> b t (h w) o 1', b=B, t=F)
        return x


def viz_attention_map(all_images, cluster_sim, obj_assignments, positive_clusters, clip_ids, save_dir="visualizations"):
    """
    Visualizes and saves concatenated images with object masks and similarity overlays.

    Args:
        all_images (torch.Tensor): Tensor of shape (B, T, H, W, 3) with uint8 values.
        cluster_sim (list of dict): Per-frame similarity per cluster.
        obj_assignments (torch.Tensor): Tensor of shape (B, H, W) with integer cluster indices.
        positive_clusters (list of list): List of lists indicating which clusters are "positive" per sample.
        save_dir (str): Directory where images will be saved.
    """
    os.makedirs(save_dir, exist_ok=True)
    all_images = all_images[:, 0]  # Take only the first frame if shape is (B, T, H, W, 3)
    B, H, W, _ = all_images.shape

    # Color palette for clusters
    colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', 
              '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3', 
              '#808000', '#ffd8b1', '#000075', '#808080', '#C56932', '#b7a58c', '#3a627d', '#9abc15', 
              '#54810c', '#a7389c', '#687253', '#61f584', '#9a17d4', '#52b0c1', '#21f5b4', '#a2856c', 
              '#9b1c34', '#4b1062', '#7cf406', '#0b1f63']
    color_cycle = cycle(colors)
    colormap = cm.get_cmap('bwr')

    for i in (range(min(B, 250))):
        image = all_images[i].cpu().numpy()

        # Normalize cluster similarity
        sorted_items = sorted(cluster_sim[i].items(), key=lambda x: x[1])
        n = len(sorted_items)

        if n == 1:
            print(f"Warning: Only one cluster for clip {clip_ids[i]}. Skipping visualization.")
            continue  # Skip if there's only one cluster

        min_val = 0.1
        max_val = 0.9
        cluster_sim[i] = {k: min_val + j * (max_val - min_val) / (n - 1) for j, (k, _) in enumerate(sorted_items)}

        unique_clusters = torch.unique(obj_assignments[i]).tolist()
        cluster_colors = {idx: hex_to_rgb(next(color_cycle)) for idx in unique_clusters}

        overlay = np.zeros((*obj_assignments[i].shape, 3), dtype=np.float32)
        sim_overlay = np.zeros((*obj_assignments[i].shape, 3), dtype=np.float32)

        for idx in unique_clusters:
            mask = obj_assignments[i].cpu().numpy() == idx
            overlay[mask] = cluster_colors[idx]
            sim_overlay[mask] = cluster_sim[i][idx]

        sim_overlay = 1 - sim_overlay
        attention_rgb = (colormap(sim_overlay[:, :, 0])[:, :, :3] * 255).astype(np.uint8)
        attention_rgb = cv2.cvtColor(attention_rgb, cv2.COLOR_RGB2BGR)
        attention_rgb[obj_assignments[i].cpu().numpy() == -1] = 0

        v_a_overlay = cv2.addWeighted(image, 0.3, attention_rgb, 0.7, 0)

        if len(positive_clusters[i]) == 0:
            print(f"Warning: No positive clusters for clip {clip_ids[i]}. Skipping visualization.")
            continue
        
        mask = obj_assignments[i].cpu().numpy() == positive_clusters[i][0]
        two_masks = False
        if len(positive_clusters[i]) > 1:
            # mask |= obj_assignments[i].cpu().numpy() == positive_clusters[i][1]
            mask2 = obj_assignments[i].cpu().numpy() == positive_clusters[i][1]
            mask2 = (mask2.astype(np.uint8) * 255)
            mask2_rgb = cv2.cvtColor(mask2, cv2.COLOR_GRAY2RGB)
            two_masks = True

        mask = (mask.astype(np.uint8) * 255)
        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

        # Ensure all images are uint8 and RGB
        # if two_masks:
        #     images_collection = [[image, mask_rgb, v_a_overlay], [image, mask2_rgb, v_a_overlay]]
        # else:
        #     images_collection = [[image, mask_rgb, v_a_overlay]]
        if two_masks:
            images_collection = [[mask_rgb, v_a_overlay], [mask2_rgb, v_a_overlay]]
        else:
            images_collection = [[mask_rgb, v_a_overlay]]
        
        for j, images in enumerate(images_collection):
            images_rgb = []
            for img in images:
                if img.dtype == np.float32 or img.dtype == np.float64:
                    img = (img * 255).clip(0, 255).astype(np.uint8)
                images_rgb.append(img)

            # Concatenate horizontally
            concat_img = np.hstack(images_rgb)  # (H, W*3, 3)

            # Save as image
            save_path = os.path.join(save_dir, f"{clip_ids[i]}_{j}.jpg")
            cv2.imwrite(save_path, cv2.cvtColor(concat_img, cv2.COLOR_RGB2BGR))

            concat_image_to_video(
                image_path=save_path,
                video_path=f"/coc/flash5/datasets/ego4d/ego4dsounds_540p_correct_shape/{clip_ids[i]}.mp4",
                output_path=os.path.join(save_dir, f"{clip_ids[i]}.mp4"),
                position="left"
            )
            break