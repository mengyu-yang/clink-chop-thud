import os
import sys
import numpy as np
import ast
sys.path.append('./AudioSpectrogramTransformer')
sys.path.append('./spot')
sys.path.append('./EgoVLP')

import argparse

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torchinfo import summary
from dataset import Ego4DDataset, Ego4D_Discovery_Eval
import wandb
from torchmetrics.functional.classification import binary_roc, binary_precision_recall_curve
from torcheval.metrics.functional.aggregation.auc import auc
from einops import rearrange

from models import SoundingObjectsModel, Aligner
from train_utils import AllGather_multi, sim_matrix

from tqdm import tqdm


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
        experiment_id = (
            self.args.slurm_job_id
            if self.args.slurm_job_id is not None
            else self.args.exp_name
        )

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
    
    
    def calculate_similarity_map(self, text_embeddings, slot_embeddings, audio_embeddings, has_negatives=False, eval=False):
        text_embeddings = F.normalize(text_embeddings, dim=-1)
        slot_embeddings = F.normalize(slot_embeddings, dim=-1)
        audio_embeddings = F.normalize(audio_embeddings, dim=-1)

        if has_negatives and not eval:
            v_a_logits = torch.einsum('nsd,md->nms', slot_embeddings, audio_embeddings) / self.args.temperature
            v_t_logits = torch.einsum('nsd,md->nms', slot_embeddings, text_embeddings) / self.args.temperature
        else:
            v_a_logits = torch.einsum('nd,md->nm', slot_embeddings, audio_embeddings) / self.args.temperature
            v_t_logits = torch.einsum('nd,md->nm', slot_embeddings, text_embeddings) / self.args.temperature
        
        a_t_logits = torch.einsum('nd,md->nm', audio_embeddings, text_embeddings) / self.args.temperature

        return v_a_logits, v_t_logits, a_t_logits
        
    
    def contrastive_loss(self, v_a_logits, v_t_logits, a_t_logits, ignore_index):
        
        # Transpose so we're computing loss symetrically
        a_v_logits = v_a_logits.permute(1, 0)  # (B, B)
        t_v_logits = v_t_logits.permute(1, 0)  # (B, B)
        t_a_logits = a_t_logits.permute(1, 0)  # (B, B)

        # Compute loss
        labels = torch.arange(v_a_logits.shape[0]).long().to(self.device)
        mask = ~torch.isin(labels, torch.tensor(list(ignore_index), device=self.device))

        v_a_loss = F.cross_entropy(v_a_logits, labels, reduction='none') + F.cross_entropy(a_v_logits, labels, reduction='none')
        v_t_loss = F.cross_entropy(v_t_logits, labels, reduction='none') + F.cross_entropy(t_v_logits, labels, reduction='none')
        a_t_loss = F.cross_entropy(a_t_logits, labels, reduction='none') + F.cross_entropy(t_a_logits, labels, reduction='none')

        # Compute average loss given the mask
        v_a_loss = v_a_loss[mask].mean()
        v_t_loss = v_t_loss[mask].mean()
        a_t_loss = a_t_loss[mask].mean()

        contrastive_loss = (v_a_loss + v_t_loss + a_t_loss) / 3

        # Store individual losses
        per_modality_losses = {"v_a_loss": v_a_loss.item(), "v_t_loss": v_t_loss.item(), "a_t_loss": a_t_loss.item()}

        # Compute accuracy
        v_a_acc = (v_a_logits.argmax(dim=1) == labels).float().mean()
        v_t_acc = (v_t_logits.argmax(dim=1) == labels).float().mean()
        a_t_acc = (a_t_logits.argmax(dim=1) == labels).float().mean()

        accuracies = {"v_a_acc": v_a_acc.item(), "v_t_acc": v_t_acc.item(), "a_t_acc": a_t_acc.item()}

        return contrastive_loss, accuracies, per_modality_losses
    

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

    
    def evaluate_soundingactions(self):
        self.model.eval()

        all_v_a_logits = []
        all_v_t_logits = []
        all_a_t_logits = []

        all_neg_v_a_logits = []
        all_neg_v_t_logits = []

        all_targets = []
        with torch.no_grad():
            for _, batch in enumerate(tqdm(self.val_loader, disable=not self.args.local_rank==0)):

                batch = self.preprocess_batch(batch)

                # Forward pass
                text_embeddings, slot_embeddings, dec_slots_attns, dec_recon, obj_mask_patchified, audio_embeddings = self.model(batch)

                text_embeddings, dec_recon, audio_embeddings = self.aligner(text_embeddings, dec_recon, audio_embeddings)

                dec_recon = rearrange(dec_recon, '(b t) n d -> b t n 1 d', t=self.args.video_num_frames)
                
                # Patch-wise multiply the object mask with the decoder reconstruction so that only embeddings of objects are non-zero
                obj_embeds = obj_mask_patchified * dec_recon  # (B, num_frames, num_patches, num_objects, D)

                neg_mask_patchified = ~obj_mask_patchified
                neg_embeds = neg_mask_patchified * dec_recon
                
                # Pool to get single embedding per object per frame
                mask = obj_embeds.abs().sum(dim=-1) > 0
                sum_values = (obj_embeds * mask.unsqueeze(-1)).sum(dim=2)
                num_nonzero_patches = mask.sum(dim=2).unsqueeze(-1).clamp(min=1)  # Avoid division by zero
                obj_embeds = sum_values / num_nonzero_patches
            
                mask = neg_embeds.abs().sum(dim=-1) > 0
                sum_values = (neg_embeds * mask.unsqueeze(-1)).sum(dim=2)
                num_nonzero_patches = mask.sum(dim=2).unsqueeze(-1).clamp(min=1)  # Avoid division by zero
                neg_embeds = sum_values / num_nonzero_patches
                
                # Pool over frames (returns (B, num_objects, D))
                obj_embeds = obj_embeds.mean(dim=1)
                neg_embeds = neg_embeds.mean(dim=1)

                # Pool over objects
                averaged_slots = obj_embeds.mean(dim=1) 
                averaged_neg_slots = neg_embeds.mean(dim=1)
                
                object_idx_masks = batch['object_idx_masks'][:, 0, :self.args.num_objects]  # The 0 index is for the first frame (we assume only 1 frame)
                invalid_indices_mask = (object_idx_masks.sum(dim=1) == 0)
                                
                if self.args.distributed:
                    text_embeddings = self.all_gather_no_grad(text_embeddings, self.args)
                    averaged_slots = self.all_gather_no_grad(averaged_slots, self.args)
                    averaged_neg_slots = self.all_gather_no_grad(averaged_neg_slots, self.args)
                    audio_embeddings = self.all_gather_no_grad(audio_embeddings, self.args)
                    targets = self.all_gather_no_grad(batch['is_sounding_action'], self.args)

                # Compute similarity map
                v_a_logits, v_t_logits, a_t_logits = self.calculate_similarity_map(text_embeddings, averaged_slots, audio_embeddings, eval=True)
                
                neg_v_a_logits, neg_v_t_logits, _ = self.calculate_similarity_map(text_embeddings, averaged_neg_slots, audio_embeddings, eval=True)
                
                # Take the diagonal of the similarity maps
                v_a_logits = v_a_logits[torch.arange(v_a_logits.shape[0]), torch.arange(v_a_logits.shape[0])]  # [B]
                v_t_logits = v_t_logits[torch.arange(v_t_logits.shape[0]), torch.arange(v_t_logits.shape[0])]  # [B]
                a_t_logits = a_t_logits[torch.arange(a_t_logits.shape[0]), torch.arange(a_t_logits.shape[0])]  # [B]

                neg_v_a_logits = neg_v_a_logits[torch.arange(neg_v_a_logits.shape[0]), torch.arange(neg_v_a_logits.shape[0])]  # [B]
                neg_v_t_logits = neg_v_t_logits[torch.arange(neg_v_t_logits.shape[0]), torch.arange(neg_v_t_logits.shape[0])]  # [B]
                
                all_v_a_logits.append(v_a_logits)
                all_v_t_logits.append(v_t_logits)
                all_a_t_logits.append(a_t_logits)

                all_neg_v_a_logits.append(neg_v_a_logits)
                all_neg_v_t_logits.append(neg_v_t_logits)

                all_targets.append(targets)

                del text_embeddings, averaged_slots, audio_embeddings, v_a_logits, v_t_logits, a_t_logits
            
            # Calculate ROC AUC and PR AUC
            v_a_logits = torch.cat(all_v_a_logits, dim=0)
            v_t_logits = torch.cat(all_v_t_logits, dim=0)
            a_t_logits = torch.cat(all_a_t_logits, dim=0)

            neg_v_a_logits = torch.cat(all_neg_v_a_logits, dim=0)
            neg_v_t_logits = torch.cat(all_neg_v_t_logits, dim=0)

            targets = torch.cat(all_targets, dim=0).long()
            
            v_a_roc = binary_roc(v_a_logits, targets)
            v_t_roc = binary_roc(v_t_logits, targets)
            a_t_roc = binary_roc(a_t_logits, targets)

            v_a_pr = binary_precision_recall_curve(v_a_logits, targets)
            v_t_pr = binary_precision_recall_curve(v_t_logits, targets)
            a_t_pr = binary_precision_recall_curve(a_t_logits, targets)

            v_a_roc_auc = auc(v_a_roc[0], v_a_roc[1], reorder=True)
            v_t_roc_auc = auc(v_t_roc[0], v_t_roc[1], reorder=True)
            a_t_roc_auc = auc(a_t_roc[0], a_t_roc[1], reorder=True)

            v_a_pr_auc = auc(v_a_pr[1], v_a_pr[0], reorder=True)
            v_t_pr_auc = auc(v_t_pr[1], v_t_pr[0], reorder=True)
            a_t_pr_auc = auc(a_t_pr[1], a_t_pr[0], reorder=True)
            
            if self.logger is not None:
                self.logger.log({"v_a_roc": v_a_roc_auc.item()})
                self.logger.log({"v_t_roc": v_t_roc_auc.item()})
                self.logger.log({"a_t_roc": a_t_roc_auc.item()})
                self.logger.log({"v_a_pr": v_a_pr_auc.item()})
                self.logger.log({"v_t_pr": v_t_pr_auc.item()})
                self.logger.log({"a_t_pr": a_t_pr_auc.item()})
            else:
                print(f"v_a_roc_auc: {v_a_roc_auc.item()}")
                print(f"v_t_roc_auc: {v_t_roc_auc.item()}")
                print(f"a_t_roc_auc: {a_t_roc_auc.item()}")
                print(f"v_a_pr_auc: {v_a_pr_auc.item()}")
                print(f"v_t_pr_auc: {v_t_pr_auc.item()}")
                print(f"a_t_pr_auc: {a_t_pr_auc.item()}")


    def consensus_loss_with_anchor(self, text_emb, video_emb, audio_emb, anchor='audio', alpha_values={'text': 1, 'video': 0.5, 'audio': 1}):
       
        output_vl = sim_matrix(video_emb, text_emb)
        output_av = sim_matrix(audio_emb, video_emb)
        output_al = sim_matrix(audio_emb, text_emb)

        if anchor == 'video':
            loss = torch.norm(output_vl - output_av, p=2)
        elif anchor == 'text':
            loss = torch.norm(output_vl - output_al, p=2)
        else:
            sim_agreement = torch.minimum(
                ((output_av + 1) / 2) ** alpha_values['video'],
                ((output_al + 1) / 2) ** alpha_values['text']
            ) * 2 - 1
            loss = torch.norm(output_al - sim_agreement, p=2) + torch.norm(output_av - sim_agreement, p=2)

        return loss


    def combined_loss(self, v_a_logits, v_t_logits, a_t_logits, text_embeddings, video_embeddings, audio_embeddings):
     
        contrastive_loss, accuracies, per_modality_losses = self.contrastive_loss(v_a_logits, v_t_logits, a_t_logits, [])
        # Add consensus loss with anchor
        text_embeddings = F.normalize(text_embeddings, dim=-1)
        video_embeddings = F.normalize(video_embeddings, dim=-1)
        audio_embeddings = F.normalize(audio_embeddings, dim=-1)

        alpha_values = {
            'text': self.args.alpha_text,
            'video': self.args.alpha_video,
            'audio': self.args.alpha_audio
        }
        
        # Compute consensus loss with each modality as anchor
        audio_consensus_loss = self.consensus_loss_with_anchor(
            text_embeddings, video_embeddings, audio_embeddings, anchor='audio', alpha_values=alpha_values)
        

        # Combine losses with weighting factor
        lambda_consensus = self.args.lambda_consensus
        total_loss = contrastive_loss + self.args.lambda_consensus * audio_consensus_loss

        accuracies.update({
            "audio_consensus_loss": audio_consensus_loss.item(),
            "contrastive_loss": contrastive_loss.item()
        })
        
        return total_loss, accuracies, per_modality_losses
    
    
    def _train_epoch(self, epoch):
                        
            self.model.train()

            if self.logger is not None:
                self.logger.log({"epoch": epoch})
            
            # Determine training stage based on epoch
            self.contrastive_align_stage = epoch < self.args.contrastive_align_epochs
            for batch_idx, batch in enumerate(tqdm(self.train_loader)):
                
                batch = self.preprocess_batch(batch)

                self.optimizer.zero_grad()

                text_embeddings, slot_embeddings, dec_slots_attns, dec_recon, obj_mask_patchified, audio_embeddings = self.model(batch)
                text_embeddings, dec_recon, audio_embeddings = self.aligner(text_embeddings, dec_recon, audio_embeddings)

                dec_recon = rearrange(dec_recon, '(b t) n d -> b t n 1 d', t=self.args.video_num_frames)
                
                # Patch-wise multiply the object mask with the decoder reconstruction so that only embeddings of objects are non-zero                
                obj_embeds = obj_mask_patchified * dec_recon  # (B, num_frames, num_patches, num_objects, D)

                # Pool to get single embedding per object per frame
                mask = obj_embeds.abs().sum(dim=-1) > 0
                sum_values = (obj_embeds * mask.unsqueeze(-1)).sum(dim=2)
                num_nonzero_patches = mask.sum(dim=2).unsqueeze(-1).clamp(min=1)  # Avoid division by zero
                obj_embeds = sum_values / num_nonzero_patches

                # Pool over frames (returns (B, num_objects, D))
                obj_embeds = obj_embeds.mean(dim=1)

                # Average over objects
                averaged_slots = obj_embeds.mean(dim=1)

                object_idx_masks = batch['object_idx_masks'][:, 0, :self.args.num_objects]  # The 0 index is for the first frame (we assume only 1 frame)
                invalid_indices_mask = (object_idx_masks.sum(dim=1) == 0)
                
                if self.args.distributed:
                    text_embeddings = AllGather_multi.apply(text_embeddings, self.args)
                    averaged_slots = AllGather_multi.apply(averaged_slots, self.args)
                    audio_embeddings = AllGather_multi.apply(audio_embeddings, self.args)
                    invalid_indices_mask = AllGather_multi.apply(invalid_indices_mask, self.args)
                
                # Compute similarity map
                v_a_logits, v_t_logits, a_t_logits = self.calculate_similarity_map(text_embeddings, averaged_slots, audio_embeddings, has_negatives=False)
                    
                # Compute loss
                if self.contrastive_align_stage:
                    invalid_indices = invalid_indices_mask.nonzero(as_tuple=True)[0].tolist()
                    loss, accuracies, per_modality_losses = self.contrastive_loss(v_a_logits, v_t_logits, a_t_logits, invalid_indices)
                else:
                    loss, accuracies, per_modality_losses = self.combined_loss(v_a_logits, v_t_logits, a_t_logits, text_embeddings, averaged_slots, audio_embeddings)
                
                if self.logger is not None:
                    self.logger.log({"train_loss": loss.item()})
                    accuracies = {f"train_{key}": val for key, val in accuracies.items()}
                    self.logger.log(accuracies)
                    losses = {f"train_{key}": val for key, val in per_modality_losses.items()}
                    self.logger.log(losses)
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
        self.evaluate_soundingactions()

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
    
    # Video
    args.add_argument('--video_num_frames', type=int, default=4)
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
    args.add_argument('--epochs', type=int, default=10)
    args.add_argument('--temperature', type=float, default=0.2)
    args.add_argument('--logger', action='store_true')
    args.add_argument('--exp_name', type=str, default="clink_chop_thud_train")

    args.add_argument("--freeze_video_layers", type=str, default="")
    args.add_argument("--freeze_audio_layers", type=str, default="0,1,2,3")
    args.add_argument("--freeze_text_layers", type=str, default="")
    args.add_argument('--freeze_ast', action='store_true')

    args.add_argument('--contrastive_align_epochs', type=int, default=5)

    args.add_argument('--alpha_text', type=float, default=1.0)
    args.add_argument('--alpha_video', type=float, default=0.5)
    args.add_argument('--alpha_audio', type=float, default=1.0)

    args.add_argument('--lambda_consensus', type=float, default=0.5)

    # Checkpoint params
    args.add_argument('--checkpoint', type=str, default=None)

    # Misc
    args.add_argument('--eval', action='store_true')
    args.add_argument('--cluster', action='store_true')

    args = args.parse_args()

    torch.manual_seed(args.seed)
    
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
    
    if "SLURM_JOB_ID" in os.environ.keys():
        args.slurm_job_id = os.environ["SLURM_JOB_ID"]
    else:
        args.slurm_job_id = None
    
    # Initialize logger
    if args.logger and args.local_rank == 0:
        proj_name = "clink_chop_thud"
        wandb.init(config=args, project=proj_name, group=args.exp_name, mode="online")
        logger = wandb.run
        logger.config.update(args)
    else:
        logger = None
    
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

    args.clips_dir = os.path.join(args.all_data_dir, "ego4d_train_clips")
    args.eval_clips_dir = os.path.join(args.all_data_dir, "ego4d_discovery_eval_clips")
    args.mask_dir = os.path.join(args.all_data_dir, "ego4d_train_masks")
    args.eval_mask_dir = os.path.join(args.all_data_dir, "ego4d_discovery_eval_masks")
    args.train_metadata_file = os.path.join(args.all_data_dir, "ego4d_train.csv")
    args.eval_metadata_file = os.path.join(args.all_data_dir, "ego4d_discovery_eval.csv")

    # Initialize dataloader
    train_dataset = Ego4DDataset(args)
    val_dataset = Ego4D_Discovery_Eval(args)

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

    # Train model
    trainer = ObjectsTrainer(model, aligner, train_loader, val_loader, optimizer, args, logger, device)
    
    if args.eval:
        trainer.eval()
    elif args.cluster:
        trainer.cluster_embeds()
    else:
        trainer.train()
    