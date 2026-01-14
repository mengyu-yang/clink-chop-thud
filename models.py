import copy
from types import SimpleNamespace

import torch
import torch.nn as nn
from einops import rearrange
from transformers import AutoTokenizer, CLIPTextModel

from AudioSpectrogramTransformer.src.models import ASTModel
from spot import SPOT
from spot_config import model_args as spot_args
from train_utils import PatchifyObjectMask


class SoundingObjectsModel(nn.Module):
    
    def __init__(self, args):
        super(SoundingObjectsModel, self).__init__()
        
        spot_args_ns = SimpleNamespace(**spot_args)
        spot_args_ns.max_tokens = int((spot_args_ns.val_image_size/16)**2)
        self.spot_args = spot_args_ns
        self.args = args
       
        encoder = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
        encoder_second = copy.deepcopy(encoder).eval()
        self.spot = SPOT(encoder, spot_args_ns, encoder_second)
        checkpoint = torch.load(spot_args_ns.checkpoint_path, map_location='cpu')
        checkpoint['model'] = {k.replace("tf_dec.", "dec."): v for k, v in checkpoint['model'].items()} # compatibility with older runs
        self.spot.load_state_dict(checkpoint['model'], strict = True)

        self.clip = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

        self.ast_model = ASTModel(
            label_dim=args.ast_output_dim, 
            input_tdim=args.ast_input_tdim, 
        )

        self.mask_patchifier = PatchifyObjectMask(
            img_size=args.video_input_res, patch_size=16, in_chans=1, num_frames=args.video_num_frames, device='cuda')
        

    def forward(self, batch):

        # Run SPOT model
        # Input shape: (B, C, H, W)
        # Put the frames in the batch dimension
        images = rearrange(batch['video'], 'b t c h w -> (b t) c h w')

        _, _, dec_slots_attns, slots, dec_recon, attn_logits, emb_input = self.spot(images)
    
        # Patchify the object masks
        obj_mask_patchified = self.mask_patchifier(batch['object_masks'])
        
        # Run CLIP model
        inputs = self.clip_tokenizer(batch['narration'], padding=True, return_tensors="pt")
        
        # To device
        for key in inputs:
            inputs[key] = inputs[key].to(images.device)
        
        clip_outputs = self.clip(**inputs)
        text_token_embeddings = clip_outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)
        text_pooled_embeddings = clip_outputs.pooler_output  # pooled (EOS token) states (batch_size, hidden_size)

        # Run AST model
        audio_embeddings = self.ast_model(batch['fbank'])
        
        # Pretrained AST model outputs dtype float16, so convert to float32
        audio_embeddings = audio_embeddings.float()

        return text_pooled_embeddings, slots, dec_slots_attns, dec_recon, obj_mask_patchified, audio_embeddings


class Aligner(nn.Module):
    def __init__(self, args):
        super(Aligner, self).__init__()
        
        spot_args_ns = SimpleNamespace(**spot_args)
        spot_args_ns.max_tokens = int((spot_args_ns.val_image_size/16)**2)
        self.spot_args = spot_args_ns
        
        obj_input_d = self.spot_args.d_model
        self.slot_aligner = nn.Sequential(
            nn.LayerNorm(obj_input_d),
            nn.Linear(obj_input_d, args.ast_output_dim),
        )

        self.text_aligner = nn.Sequential(
            nn.LayerNorm(args.clip_hidden_size),
            nn.Linear(args.clip_hidden_size, args.ast_output_dim),
        )

        self.audio_aligner = nn.Sequential(
            nn.LayerNorm(args.ast_output_dim),
            nn.Linear(args.ast_output_dim, args.ast_output_dim),
        )
    
    def forward(self, text_embeddings, slot_embeddings, audio_embeddings):        
        # Run alignment layers
        slot_embeddings = self.slot_aligner(slot_embeddings)
        text_embeddings = self.text_aligner(text_embeddings)
        audio_embeddings = self.audio_aligner(audio_embeddings)
        
        return text_embeddings, slot_embeddings, audio_embeddings
