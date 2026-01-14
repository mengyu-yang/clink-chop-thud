import os
import sys
import random
import numpy as np
import ast
sys.path.append('./AudioSpectrogramTransformer')
sys.path.append('./spot')
sys.path.append('./EgoVLP')

import argparse
import transformers
from collections import defaultdict

import torch
import torch.nn.functional as F
import torchvision
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torchinfo import summary
from dataset import Ego4DDataset, Ego4D_Detection_Eval, EpicKitchensDataset, EpicKitchens_Detection_Eval
from einops import rearrange
import wandb

from models import SoundingObjectsModel, Aligner
from train_utils import AllGather_multi, viz_attention_map

from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

torch.manual_seed(19)


class ObjectsTrainer():

    def __init__(self, model, aligner, train_loader, val_loader, optimizer, args, logger, device):
        
        self.model = model
        self.aligner = aligner
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.args = args
        self.logger = logger
        self.device = device
        self.start_epoch = 0
        self.global_step = 0

        # Create experiment directory
        experiment_id = self.args.exp_name

        if self.logger is not None:
            self.experiment_dir = f"{wandb.run.project}/{experiment_id}"
        else:
            self.experiment_dir = f"checkpoints/{experiment_id}"
        
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        # Check for explicitly defined model checkpoint
        if args.checkpoint is not None:
            if args.local_rank == 0:
                checkpoint = torch.load(args.checkpoint, map_location=self.device, weights_only=False)
            else:
                checkpoint = None
            dist.barrier()
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=True)
            self.aligner.load_state_dict(checkpoint['aligner_state_dict'], strict=True)
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.start_epoch = checkpoint['epoch']
            self.global_step = checkpoint['global_step']
            print("Loaded model from checkpoint")
            dist.barrier()


    def calculate_similarity_map(self, text_embeddings, slot_embeddings, audio_embeddings, has_negatives=False):
        text_embeddings = F.normalize(text_embeddings, dim=-1)
        slot_embeddings = F.normalize(slot_embeddings, dim=-1)
        audio_embeddings = F.normalize(audio_embeddings, dim=-1)

        if has_negatives:
            v_a_logits = torch.einsum('nsd,md->nms', slot_embeddings, audio_embeddings) / self.args.temperature
            v_t_logits = torch.einsum('nsd,md->nms', slot_embeddings, text_embeddings) / self.args.temperature
        else:
            # Compute similarity between all pairs
            # text_embeddings: [B, D]
            # slot_embeddings: [B, D]
            # audio_embeddings: [B, D]
            v_a_logits = torch.einsum('nd,md->nm', slot_embeddings, audio_embeddings) / self.args.temperature
            v_t_logits = torch.einsum('nd,md->nm', slot_embeddings, text_embeddings) / self.args.temperature
        
        a_t_logits = torch.einsum('nd,md->nm', audio_embeddings, text_embeddings) / self.args.temperature

        return v_a_logits, v_t_logits, a_t_logits
        
    
    def contrastive_loss(self, v_a_logits, ignore_index, has_negatives=False):
        
        if has_negatives:
            # Separate out hard negatives into separate tensor
            v_a_logits, v_a_hard_negatives = v_a_logits[:, :, 0], v_a_logits[:, :, 1]

        # Transpose so we're computing loss symetrically
        a_v_logits = v_a_logits.permute(1, 0)  # (B, B)

        if has_negatives:
            a_v_hard_negatives = v_a_hard_negatives.permute(1, 0)

            # Concatenate logits 
            v_a_logits = torch.cat([v_a_logits, v_a_hard_negatives], dim=-1)
            a_v_logits = torch.cat([a_v_logits, a_v_hard_negatives], dim=-1)
        
        # Compute loss
        labels = torch.arange(v_a_logits.shape[0]).long().to(self.device)
        mask = ~torch.isin(labels, torch.tensor(list(ignore_index), device=self.device))

        v_a_loss = F.cross_entropy(v_a_logits, labels, reduction='none') + F.cross_entropy(a_v_logits, labels, reduction='none')

        # Compute average loss given the mask
        v_a_loss = v_a_loss[mask].mean()

        contrastive_loss = v_a_loss

        return contrastive_loss
    

    def preprocess_batch(self, batch):
        # Move tensors to device
        for key, value in batch.items():
            # If the value is a tensor
            if isinstance(value, torch.Tensor):
                batch[key] = batch[key].to(device)
        
        # Move video_np to numpy
        batch['video_np'] = batch['video_np'].cpu().numpy()
        
        if len(batch['object_masks'].shape) == 4:
            batch['object_masks'] = batch['object_masks'].unsqueeze(1)
        
        return batch
    
    
    def all_gather_no_grad(self, tensor, args):
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        output = [torch.empty_like(tensor) for _ in range(args.world_size)]
        dist.all_gather(output, tensor)
        return torch.cat(output, 0)


    def calculate_cluster_averages_per_example(self, cluster_indices, cosine_similarities, mode='mean'):
        """Calculate average cosine similarity per cluster for each batch example"""
        batch_results = []
        
        for b in range(cluster_indices.size(0)):  # Iterate through batch dimension
            # Flatten tensors for current example
            clusters = cluster_indices[b].flatten()
            similarities = cosine_similarities[b].flatten()
            
            # Calculate cluster means using vectorized operations
            if mode == 'mean':
                example_means = {
                    int(cluster.item()): similarities[clusters == cluster].mean().item()
                    for cluster in torch.unique(clusters)
                }
            elif mode == 'max':
                example_means = {
                    int(cluster.item()): similarities[clusters == cluster].max().item()
                    for cluster in torch.unique(clusters)
                }
            
            batch_results.append(example_means)
        
        return batch_results


    def filter_and_merge_masks(
        self,
        masks_orig: torch.Tensor,           # (B, N1, H, W)
        object_masks: torch.Tensor,         # (B, N2, H, W)
        iou_threshold: float = 0.6,
        min_area: int = 150
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Filter and merge masks, returning additional metadata about origin and emptiness.

        Args:
            masks_orig (Tensor): Binary masks of shape (B, N1, H, W)
            object_masks (Tensor): Binary masks of shape (B, N2, H, W)
            iou_threshold (float): IoU threshold for overlap suppression
            min_area (int): Minimum pixel count for a mask to be retained

        Returns:
            final_masks (Tensor): Combined masks of shape (B, N, H, W)
            active_mask (Tensor): Binary tensor of shape (B, N), 1 = non-empty mask
            source_mask (Tensor): Binary tensor of shape (B, N), 1 = from object_masks, 0 = from masks_orig
        """
        B, N1, H, W = masks_orig.shape
        N2 = object_masks.shape[1]
        device = masks_orig.device

        final_masks_batch = []
        active_mask_batch = []
        source_mask_batch = []

        for b in range(B):
            masks_b = masks_orig[b]
            kept = []
            sources = []

            # Area + IoU filter for original masks
            for i in range(N1):
                mask_i = masks_b[i]
                if mask_i.sum() < min_area:
                    continue
                keep = True
                for j in range(len(kept)):
                    existing = kept[j]
                    intersection = (mask_i & existing).sum()
                    union = mask_i.sum() + existing.sum() - intersection
                    iou = intersection.float() / (union.float() + 1e-6)
                    if iou > iou_threshold:
                        keep = False
                        break
                if keep:
                    kept.append(mask_i)
                    sources.append(0)  # from masks_orig

            # Add object_masks if no overlap
            positive_obj_idx = []
            for i in range(N2):
                mask_obj = object_masks[b, i]

                # Check if mask_obj is empty
                if mask_obj.sum() == 0:
                    continue

                keep = True
                for j, existing in enumerate(kept):
                    intersection = (mask_obj & existing).sum()
                    union = mask_obj.sum() + existing.sum() - intersection
                    iou = intersection.float() / (union.float() + 1e-6)
                    if iou > iou_threshold:
                        keep = False
                        positive_obj_idx.append(j)
                        break
                if keep:
                    kept.append(mask_obj)
                    sources.append(1)  # from object_masks
                    positive_obj_idx.append(len(kept) - 1)

            if kept:
                kept_tensor = torch.stack(kept, dim=0)
                src_tensor = torch.tensor(sources, dtype=torch.uint8, device=device)
                act_tensor = torch.ones_like(src_tensor)
            else:
                kept_tensor = torch.zeros((0, H, W), dtype=masks_orig.dtype, device=device)
                src_tensor = torch.zeros((0,), dtype=torch.uint8, device=device)
                act_tensor = torch.zeros((0,), dtype=torch.uint8, device=device)
            
            final_masks_batch.append(kept_tensor)
            source_mask_batch.append(positive_obj_idx)
            active_mask_batch.append(act_tensor)

        # Pad to max length
        max_n = max(m.shape[0] for m in final_masks_batch)
        final_masks = torch.zeros((B, max_n, H, W), dtype=masks_orig.dtype, device=device)
        source_mask = torch.zeros((B, max_n), dtype=torch.uint8, device=device)
        active_mask = torch.zeros((B, max_n), dtype=torch.uint8, device=device)

        for b in range(B):
            n = final_masks_batch[b].shape[0]
            final_masks[b, :n] = final_masks_batch[b]
            active_mask[b, :n] = active_mask_batch[b]

            pos_idx = source_mask_batch[b]
            # Set source_mask to 1 for positive object indices
            source_mask[b, pos_idx] = 1

        return final_masks, active_mask, source_mask

    
    def evaluate_scene_object_retrieval_pooled_objects(self, num_eval_epochs=1):
         
        self.model.eval()
        self.model.module.spot.training = False  
        
        with torch.no_grad():
            all_matches_mean = []    
            for _, batch in enumerate(tqdm(self.val_loader, disable=not self.args.local_rank==0)):

                batch = self.preprocess_batch(batch)

                # Forward pass
                text_embeddings, slot_embeddings, dec_slots_attns, dec_recon, obj_mask_patchified, audio_embeddings = self.model(batch)

                object_pool_masks = batch['object_pool_masks']  # (B, num_pool_objects, H, W)
                object_pool_idx_masks = batch['object_pool_idx_masks']  # (B, num_pool_objects)

                object_masks = batch['object_masks'][:, 0]  # (B, num_objects, H, W)
                object_idx_masks = batch['object_idx_masks'][:, 0]  # (B, num_objects)

                object_pool_masks_filtered, object_idx_mask, positive_mask = self.filter_and_merge_masks(object_pool_masks, object_masks, iou_threshold=0.6, min_area=150)

                B, N, H, W = object_pool_masks_filtered.shape
                obj_assignments = torch.full((B, H, W), fill_value=-1, dtype=torch.long, device=object_pool_masks_filtered.device)

                for b in range(B):
                    mask_b = object_pool_masks_filtered[b]  # (N, H, W)
                    areas = mask_b.view(N, -1).sum(dim=1)  # (N,)

                    sorted_indices = torch.argsort(areas, descending=True)
                    for idx, mask_idx in enumerate(sorted_indices):
                        mask = mask_b[mask_idx].bool()
                        obj_assignments[b][mask] = mask_idx.item() 
                
                text_embeddings, dec_recon, audio_embeddings = self.aligner(text_embeddings, dec_recon, audio_embeddings)
                
                if self.args.distributed:
                    text_embeddings = self.all_gather_no_grad(text_embeddings, self.args)
                    image_embeddings = self.all_gather_no_grad(dec_recon, self.args)    
                    audio_embeddings = self.all_gather_no_grad(audio_embeddings, self.args)
                    is_sounding_action = self.all_gather_no_grad(batch['is_sounding_action'], self.args)
                    images = self.all_gather_no_grad(torch.from_numpy(batch['video_np']).to(self.device), self.args)
                    obj_assignments = self.all_gather_no_grad(obj_assignments, self.args)
                    positive_mask = self.all_gather_no_grad(positive_mask, self.args)

                sounding_action_indices = is_sounding_action.nonzero(as_tuple=True)[0]
                images = images[sounding_action_indices]
                image_embeddings = image_embeddings[sounding_action_indices]
                audio_embeddings = audio_embeddings[sounding_action_indices]
                obj_assignments = obj_assignments[sounding_action_indices]
                positive_mask = positive_mask[sounding_action_indices]
                clip_ids = [batch['clip_id'][i] for i in sounding_action_indices]

                # Calculate similarity map
                v_a_sim = torch.einsum('nsd,md->nms', image_embeddings, audio_embeddings)

                # Take samples along the diagonal
                v_a_sim = v_a_sim[torch.arange(v_a_sim.shape[0]), torch.arange(v_a_sim.shape[0])]  # (B, num_patches)

                min_vals = v_a_sim.min(dim=1, keepdim=True).values  # Min along each row
                max_vals = v_a_sim.max(dim=1, keepdim=True).values  # Max along each row
                v_a_sim = (v_a_sim - min_vals) / (max_vals - min_vals + 1e-8)  # Avoid division by zero
                
                # Reshape to (B, H, W)
                v_a_sim = v_a_sim.reshape(-1, self.args.video_input_res // 16, self.args.video_input_res // 16)

                # Interpolate to original resolution
                v_a_sim = F.interpolate(v_a_sim.unsqueeze(1), size=self.args.video_input_res, mode='bilinear').squeeze(1)

                cluster_sim_mean = self.calculate_cluster_averages_per_example(obj_assignments, v_a_sim)
                
                # Convert multi-hot positive_mask to integer indices
                positive_idx = [row.nonzero(as_tuple=True)[0].tolist() for row in positive_mask]

                highest_avg_clusters_mean = [max(ex, key=ex.get) for ex in cluster_sim_mean]

                matches_mean = [avg in best for avg, best in zip(highest_avg_clusters_mean, positive_idx)]
                
                all_matches_mean.extend(matches_mean)

                # viz_attention_map(images, cluster_sim_mean, obj_assignments, positive_idx, clip_ids, save_dir='object_det_viz_ego4d')
            
            success_rate_mean = sum(all_matches_mean) / len(all_matches_mean)

            if self.logger is not None:
                self.logger.log({"success_rate_mean": success_rate_mean})
            else:
                print(f"Mean top-1 success rate: {round(success_rate_mean * 100, 1)}%")
            
        self.model.module.spot.training = True     

    
    def _train_epoch(self, epoch):
            
            self.model.train()

            if self.logger is not None:
                self.logger.log({"epoch": epoch})

            # Determine training stage based on epoch
            for batch_idx, batch in enumerate(tqdm(self.train_loader)):
                
                batch = self.preprocess_batch(batch)

                self.optimizer.zero_grad()

                text_embeddings, slot_embeddings, dec_slots_attns, dec_recon, obj_mask_patchified, audio_embeddings = self.model(batch)

                text_embeddings, dec_recon, audio_embeddings = self.aligner(text_embeddings, dec_recon, audio_embeddings)

                dec_recon = rearrange(dec_recon, '(b t) n d -> b t n 1 d', t=self.args.video_num_frames)
                # Patch-wise multiply the object mask with the decoder reconstruction so that only embeddings of objects are non-zero
                obj_embeds = obj_mask_patchified * dec_recon  # (B, num_frames, num_patches, num_objects, D)
                
                # Subsample negative mask
                neg_mask_patchified = ~obj_mask_patchified
                true_indices = neg_mask_patchified.nonzero(as_tuple=False)
                num_true = true_indices.shape[0]  # Get the number of True elements
                num_to_flip = int((1-self.args.inverse_neg_samplerate) * num_true)  # Determine how many to flip
                indices_to_flip = torch.randperm(num_true)[:num_to_flip]
                selected_indices = tuple(true_indices[indices_to_flip].T)
                neg_mask_patchified[selected_indices] = False

                neg_embeds = neg_mask_patchified * dec_recon
                obj_embeds = torch.cat([obj_embeds, neg_embeds], dim=3)

                # Pool to get single embedding per object per frame
                mask = obj_embeds.abs().sum(dim=-1) > 0
                sum_values = (obj_embeds * mask.unsqueeze(-1)).sum(dim=2)
                num_nonzero_patches = mask.sum(dim=2).unsqueeze(-1).clamp(min=1)  # Avoid division by zero
                obj_embeds = sum_values / num_nonzero_patches

                # Pool over frames (returns (B, num_objects, D))
                obj_embeds = obj_embeds.mean(dim=1)

                # Average over objects
                obj_embeds, obj_embeds_negative = obj_embeds[:, :self.args.num_objects], obj_embeds[:, self.args.num_objects:]
                obj_embeds = obj_embeds.mean(dim=1, keepdim=True)
                obj_embeds_negative = obj_embeds_negative.mean(dim=1, keepdim=True)
                averaged_slots = torch.cat([obj_embeds, obj_embeds_negative], dim=1)

                object_idx_masks = batch['object_idx_masks'][:, 0, :self.args.num_objects]  # The 0 index is for the first frame (we assume only 1 frame)
                invalid_indices_mask = (object_idx_masks.sum(dim=1) == 0)

                if self.args.distributed:
                    text_embeddings = AllGather_multi.apply(text_embeddings, self.args)
                    averaged_slots = AllGather_multi.apply(averaged_slots, self.args)
                    slot_embeddings = AllGather_multi.apply(slot_embeddings, self.args)
                    audio_embeddings = AllGather_multi.apply(audio_embeddings, self.args)
                    invalid_indices_mask = AllGather_multi.apply(invalid_indices_mask, self.args)
                
                # Compute similarity map
                v_a_logits, _, _ = self.calculate_similarity_map(text_embeddings, averaged_slots, audio_embeddings, has_negatives=True)
                    
                # Compute loss
                invalid_indices = invalid_indices_mask.nonzero(as_tuple=True)[0].tolist()
                loss = self.contrastive_loss(v_a_logits, invalid_indices, has_negatives=True)

                if self.logger is not None:
                    self.logger.log({"train_loss": loss.item()})
                    self.logger.log({"lr": self.optimizer.param_groups[0]['lr']})
                    self.logger.log({"global_step": self.global_step})
                
                loss.backward()
                self.optimizer.step()

                self.global_step += 1             

            # Evaluate model
            self.eval()

            # Save model
            if epoch % 1 == 0 and self.args.local_rank == 0:
                save_dict = {
                    "model_state_dict": self.model.state_dict(),
                    "aligner_state_dict": self.aligner.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "epoch": epoch,
                    "global_step": self.global_step,
                    "args": self.args
                }
                torch.save(save_dict, os.path.join(self.experiment_dir, f"objects_model_latest.pth"))
            torch.distributed.barrier()
                
                
    def train(self):
        for epoch in range(self.start_epoch + 1, self.args.epochs, 1):
            if self.args.local_rank == 0:
                print(f"########## Epoch {epoch} ##########")
            self._train_epoch(epoch)
    

    def eval(self):
        self.evaluate_scene_object_retrieval_pooled_objects()


if __name__ == '__main__':
    
    try:    # with ddp
        master_address = os.environ['MASTER_ADDR']
        master_port = int(os.environ['MASTER_PORT'])
        world_size = int(os.environ['WORLD_SIZE'])
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        distributed = True
        if local_rank == 0:
            print(f"Using DistributedDataParallel with {torch.cuda.device_count()} GPUs")
        print(f"The rank(local) of this node is {local_rank}")

    except:  # for debug only
        master_address = 9339
        master_port = 1
        world_size = 1
        rank = 0
        local_rank = 0
        distributed = False
    
    args = argparse.ArgumentParser()

    # Data path
    args.add_argument('--all_data_dir', type=str, required=True)
    
    # Video model params
    args.add_argument('--video_num_frames', type=int, default=1)
    args.add_argument('--video_input_res', type=int, default=224)
    args.add_argument('--num_objects', type=int, default=2)

    # Audio
    args.add_argument('--audio_num_secs', type=float, default=1.5)
    args.add_argument('--audio_sample_rate', type=int, default=16000)
    args.add_argument('--num_mel_bins', type=int, default=128)

    # Text model params
    args.add_argument('--clip_hidden_size', type=int, default=512)

    # AST model params
    args.add_argument('--ast_output_dim', type=int, default=256)
   
    # Data params
    args.add_argument('--dataset', type=str, choices=["ego4d", "epic_kitchens"])

    # Data parallel
    args.add_argument('-k', '--local-rank', type=int, default=local_rank)
    args.add_argument('-ma', '--master_address', default=master_address)
    args.add_argument('-mp', '--master_port', type=int, default=master_port)
    args.add_argument('-ws', '--world_size', type=int, default=world_size)
    args.add_argument('-rk', '--rank', type=int, default=rank)
    args.add_argument('--distributed', type=bool, default=distributed)

    # Training params
    args.add_argument('--seed', type=int, default=0)
    args.add_argument('--batch_size', type=int, default=32)
    args.add_argument('--lr', type=float, default=5e-5)
    args.add_argument('--epochs', type=int, default=16)
    args.add_argument('--temperature', type=float, default=0.2)
    args.add_argument('--logger', action='store_true')
    args.add_argument('--exp_name', type=str, default="sounding_object_detection")

    args.add_argument("--freeze_video_layers", type=str, default="")
    args.add_argument("--freeze_audio_layers", type=str, default="")
    args.add_argument("--freeze_text_layers", type=str, default="")
    args.add_argument('--freeze_ast', action='store_true')

    args.add_argument('--inverse_neg_samplerate', type=float, default=0.5)

    args.add_argument('--alpha_text', type=float, default=1.0)
    args.add_argument('--alpha_video', type=float, default=0.5)
    args.add_argument('--alpha_audio', type=float, default=1.0)

    # Checkpoint params
    args.add_argument('--checkpoint', type=str, default=None)

    # Misc
    args.add_argument('--eval', action='store_true')

    # config = ConfigParser(args)
    args = args.parse_args()
    
    freeze_video_layers = [int(i) for i in args.freeze_video_layers.split(",")] if args.freeze_video_layers else []
    freeze_audio_layers = [int(i) for i in args.freeze_audio_layers.split(",")] if args.freeze_audio_layers else [] 
    freeze_text_layers = [int(i) for i in args.freeze_text_layers.split(",")] if args.freeze_text_layers else []

    args.ast_input_tdim = int(100 * args.audio_num_secs)
    
    torch.cuda.set_device(args.local_rank)
    
    if args.distributed:
        # DistributedDataParallel
        torch.distributed.init_process_group(backend='nccl',
                                                 init_method='tcp://{}:{}'.format(
                                                 args.master_address, args.master_port),
                                             rank=args.local_rank, world_size=args.world_size)
    
    device = torch.device(f'cuda:{args.local_rank}' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model
    model = SoundingObjectsModel(args)
    model = model.to(device)

    aligner = Aligner(args)
    aligner = aligner.to(device)
    
    # Freeze certain layers
    for name, param in model.named_parameters():
        # Freeze the slot attention model's encoders, only finetuning the decoder
        if "spot.encoder" in name:
            param.requires_grad = False
        if "spot.second_encoder" in name:
            param.requires_grad = False
        
        # Keep attention slots frozen
        if "spot.slot_attn" in name:
            param.requires_grad = False

        # Freeze text encoder
        if "clip.text_model" in name:
            param.requires_grad = False

        if args.freeze_ast:
            if "ast_model" in name:
                param.requires_grad = False
        else:
            if any(f"ast_model.v.blocks.{i}." in name for i in freeze_audio_layers):
                param.requires_grad = False
        
    # Print model summary
    if args.local_rank == 0:
        summary(model)
    
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=True)

    # Initialize dataloader
    if args.dataset == 'ego4d':
        
        args.clips_dir = os.path.join(args.all_data_dir, 'ego4d_train_clips')
        args.mask_dir = os.path.join(args.all_data_dir, 'ego4d_train_masks')
        args.train_metadata_file = os.path.join(args.all_data_dir, 'ego4d_train.csv')

        args.eval_clips_dir = os.path.join(args.all_data_dir, 'ego4d_detection_eval_clips')
        args.eval_mask_dir = os.path.join(args.all_data_dir, 'ego4d_detection_eval_masks')
        args.eval_pool_mask_dir = os.path.join(args.all_data_dir, 'ego4d_detection_eval_object_pool_masks')
        args.eval_metadata_file = os.path.join(args.all_data_dir, 'ego4d_detection_eval.csv')
        
        train_dataset = Ego4DDataset(args)
        val_dataset = Ego4D_Detection_Eval(args)

    else:

        args.clips_dir = os.path.join(args.all_data_dir, 'epickitchens_clips')
        args.mask_dir = os.path.join(args.all_data_dir, 'epickitchens_train_masks')
        args.train_metadata_file = os.path.join(args.all_data_dir, 'epickitchens_train.csv')

        args.eval_clips_dir = os.path.join(args.all_data_dir, 'epickitchens_clips')
        args.eval_mask_dir = os.path.join(args.all_data_dir, 'epickitchens_detection_eval_masks')
        args.eval_pool_mask_dir = os.path.join(args.all_data_dir, 'epickitchens_detection_eval_object_pool_masks')
        args.eval_metadata_file = os.path.join(args.all_data_dir, 'epickitchens_detection_eval.csv')

        train_dataset = EpicKitchensDataset(args)
        val_dataset = EpicKitchens_Detection_Eval(args)

    if args.distributed:
        train_sampler = DistributedSampler(train_dataset)
        val_sampler = DistributedSampler(val_dataset)

    else:
        train_sampler = None
        val_sampler = None
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None), 
                                               num_workers=13, pin_memory=True, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                                             num_workers=13, pin_memory=True, sampler=val_sampler)
    
    # Initialize optimizer
    params = [
        *model.parameters(),
        *aligner.parameters(),
        ]
    
    optimizer = torch.optim.Adam(params, lr=args.lr)

    # Initialize logger
    if args.logger and args.local_rank == 0:
        proj_name = "objects_project_localization"
        wandb.init(config=args, project=proj_name, group=args.exp_name, mode="online")
        logger = wandb.run
        logger.config.update(args)
    else:
        logger = None

    # Train model
    trainer = ObjectsTrainer(model, aligner, train_loader, val_loader, optimizer, args, logger, device)
    
    if args.eval:
        trainer.eval()
    else:
        trainer.train()
