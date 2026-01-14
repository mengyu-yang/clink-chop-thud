"""
This file defines the Ego4DSounds Dataset for loading and processing video and audio. It includes functionality
preprocessing and loading media files and debugging outputs for analysis.
"""
import os
import sys
import json
import pandas as pd
import random
import torch
import glob
from collections import defaultdict
import ast
import time
import math

from torch.utils.data import Dataset
import torch
import torchaudio
import decord
from PIL import Image
from decord import AudioReader, VideoReader
from decord import cpu, gpu
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
import logging
import types
from collections import defaultdict

decord.bridge.set_bridge('torch')


def load_npz_mask(loaded):
    
    data = {"object_masks": {}, "object_idx_masks": {}}

    # --- Load object_idx_masks ---
    if "frame_indices" in loaded and "object_idx_masks" in loaded:
        frame_indices = loaded["frame_indices"]
        idx_array = loaded["object_idx_masks"]
        for frame_i, frame_idx in enumerate(frame_indices):
            data["object_idx_masks"][int(frame_idx)] = idx_array[frame_i].tolist()

    # --- Load object_masks ---
    mask_keys = [
        k for k in loaded.files
        if k.startswith("mask_") and not k.endswith(("_shape", "_dtype"))
    ]
    for key in mask_keys:
        _, frame_idx, obj_i = key.split("_")
        frame_idx, obj_i = int(frame_idx), int(obj_i)

        shape = tuple(loaded[f"{key}_shape"])
        dtype = np.dtype(loaded[f"{key}_dtype"].item())

        unpacked = np.unpackbits(loaded[key])[: np.prod(shape)].reshape(shape)
        data["object_masks"].setdefault(frame_idx, []).append(unpacked.astype(dtype))

    # --- Load any extra metadata keys ---
    for k in loaded.files:
        if (
            k.startswith("mask_")
            or k in ["object_idx_masks", "frame_indices"]
            or k.endswith(("_shape", "_dtype"))
        ):
            continue
        # Convert NumPy object array back to Python type
        arr = loaded[k]
        if arr.shape == ():  # scalar
            data[k] = arr.item()
        else:
            data[k] = arr.tolist()

    return data


class Ego4DDataset(Dataset):
    """Dataset class for handling video and audio data from the Ego4D dataset.
    This class supports loading and preprocessing of video and audio clips based on metadata.
    """

    def __init__(self,
                args,
                ):
       
        self.args = args
        self.data_dir = os.path.expandvars(self.args.clips_dir)  # check for environment variables

        metadata_file = self.args.train_metadata_file

        self.metadata = pd.read_csv(metadata_file, sep='\t', on_bad_lines='warn')

        self.noun_dim = 582  # num of nouns of ego4d taxonomy dictionary
        self.verb_dim = 118  # num of verbs of ego4d taxonomy dictionary

        self.seed = args.seed
        self.video_transform = transforms.Compose([
            transforms.Resize(args.video_input_res),
            transforms.CenterCrop(args.video_input_res),
        ])
        self.video_normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
    def get_id(self, sample):
        if 'narration_source' in sample and 'narration_ind' in sample:
            return sample['video_uid'] + '_' + sample['narration_source'] + '_' + str(sample['narration_ind'])
        else:
            return sample['video_uid']

    def __len__(self):
        return len(self.metadata)

    @property
    def video_size(self):
        return (self.args.video_num_frames, self.args.video_input_res, self.args.video_input_res, 3)

    @property
    def spec_size(self):
        return (self.args.num_mel_bins, self.args.ast_input_tdim)

    @property
    def waveform_size(self):
        return (1, int(self.args.audio_sample_rate * self.args.audio_num_secs))

    def __getitem__(self, item):
        item = item % len(self.metadata)
        sample = self.metadata.iloc[item]
        video_fp = os.path.join(self.data_dir, sample['clip_file'])
        # logging.info(f"loading video {video_fp}")
        text = sample['clip_text']
        clip_id = self.get_id(sample)

        noun_vec, verb_vec = self.get_noun_verb(sample)

        # Load video
        video_norm, video_unnorm, frame_indices = self.load_video(video_fp, num_frames=self.args.video_num_frames)
        
        # Convert video_unnorm back to numpy and dtype uint8
        video_unnorm = video_unnorm.permute(0, 2, 3, 1)
        video_unnorm = (video_unnorm * 255).to(torch.uint8)

        # Load mask
        object_masks, object_idx_masks = self.load_mask(clip_id, frame_indices)

        # Load audio
        waveform = self.load_audio(video_fp)
        
        # Extract fbank features of shape (frames, num_mel_bins)
        fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=self.args.audio_sample_rate, use_energy=False, window_type='hanning', num_mel_bins=self.args.num_mel_bins, dither=0.0, frame_shift=10)
        
        # normalize
        fbank = (fbank + 4.26) / (4.57 * 2)
        
        # Pad or cut if necessary
        n_frames = fbank.shape[0]

        p = self.args.ast_input_tdim - n_frames

        # cut and pad
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[0:self.args.ast_input_tdim, :]
   
        return {'video': video_norm, 'video_np': video_unnorm, 'wav': waveform, 'fbank': fbank, 'narration': text, 'clip_id': clip_id,
                'object_masks': object_masks, 'object_idx_masks': object_idx_masks, 'noun_vec': noun_vec, 'verb_vec': verb_vec}
    
    
    def get_noun_verb(self, sample):
        noun_vec = torch.zeros(self.noun_dim)
        verb_vec = torch.zeros(self.verb_dim)
        noun_idx = eval(sample['tag_noun'])
        verb_idx = eval(sample['tag_verb'])
        for i in noun_idx:
            noun_vec[i] = 1
        for i in verb_idx:
            verb_vec[i] = 1

        return noun_vec, verb_vec
    
    
    def load_mask(self, clip_id, frame_indices):
        mask_path = os.path.join(self.args.mask_dir, f"{clip_id}.npz")
        loaded = np.load(mask_path, allow_pickle=True)
        mask_dict = load_npz_mask(loaded)
        
        # Get object masks for corresponding frame indices    
        all_object_masks = mask_dict['object_masks']
        object_masks = [all_object_masks[int(frame)] for frame in frame_indices]
        object_masks = np.array(object_masks)
        object_masks = torch.tensor(object_masks)    
        object_masks = self.video_transform(object_masks)

        # Get mask indices for object masks
        all_object_idx_masks = mask_dict['object_idx_masks']
        object_idx_masks = [all_object_idx_masks[int(frame)] for frame in frame_indices]
        object_idx_masks = np.array(object_idx_masks)
        
        return object_masks, object_idx_masks
    
    def load_video(self, video_fp, num_frames):
        try:
            vr = VideoReader(video_fp.replace('.wav', '.mp4'), ctx=cpu(0))

            # Get total number of frames
            total_frames = len(vr)
            all_frame_indices = np.arange(total_frames)

            # We subsample the framerate by 3
            all_frame_indices = all_frame_indices[::3]

            # Set the start index to be 1/3 of the total frames (t=1s) and the end index to be 5/6 of the total frames (t=2.5s)
            start_frame = int(total_frames / 3)
            end_frame = int(2.5 * total_frames / 3)  # 2.5s corresponds to 5/6 of the frames

            all_frame_indices = all_frame_indices[(all_frame_indices >= start_frame) & (all_frame_indices <= end_frame)]
            
            # Randomly sample num_frames frames
            frame_indices = np.random.choice(all_frame_indices, num_frames, replace=False)
            frame_indices.sort()
            imgs = vr.get_batch(frame_indices)

        except Exception as e:
            print('failed to load video, use black image instead', e)
            imgs = torch.zeros(self.video_size)
        
        imgs = (imgs / 255.0).permute(0, 3, 1, 2)  # [T, H, W, C] ---> [T, C, H, W]
        imgs = self.video_transform(imgs)
        imgs_norm = self.video_normalize(imgs)
        return imgs_norm, imgs, frame_indices

    def load_audio(self, audio_fp):
        try:
            ar = AudioReader(audio_fp.replace('.wav', '.mp4'), ctx=cpu(0), sample_rate=self.args.audio_sample_rate)
            waveform = ar[:]
            
            start_idx = int(self.args.audio_sample_rate * 1)  # start at 1s
            end_idx = int(self.args.audio_sample_rate * 2.5)  # end at 2.5s
            waveform = waveform[:, start_idx:end_idx]
        except Exception as e:
            print(f'Exception while reading audio file {audio_fp} with {e}')
            waveform = torch.zeros(self.waveform_size)
            
        return waveform


class Ego4D_Discovery_Eval(Ego4DDataset):
    """Dataset class for handling video and audio data from the Ego4D dataset.
    This class supports loading and preprocessing of video and audio clips based on metadata.
    """

    def __init__(self,
                args,
                ):
       
        self.args = args
        self.data_dir = os.path.expandvars(self.args.eval_clips_dir)  # check for environment variables

        self.noun_dim = 582  # num of nouns of ego4d taxonomy dictionary
        self.verb_dim = 118  # num of verbs of ego4d taxonomy dictionary
       
        self.metadata = pd.read_csv(args.eval_metadata_file, sep='\t', on_bad_lines='warn')

        self.seed = args.seed
        self.video_transform = transforms.Compose([
            transforms.Resize(args.video_input_res),
            transforms.CenterCrop(args.video_input_res),
        ])
        self.video_normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
    def __getitem__(self, item):
        item = item % len(self.metadata)
        sample = self.metadata.iloc[item]
        narration_time = math.ceil(sample["narration_time"])
        clip_id = f"{sample['video_uid']}_{narration_time}"
        video_fp = os.path.join(self.data_dir, f"{clip_id}.mp4")
        # logging.info(f"loading video {video_fp}")
        text = sample['clip_text']
        is_sounding_action = sample['positive']
        if 'action_group' in sample:
            action_group = sample['action_group']
        else:
            action_group = -1

        noun_vec, verb_vec = self.get_noun_verb(sample)

        # Load video
        video_norm, video_unnorm, frame_indices = self.load_video(video_fp, num_frames=self.args.video_num_frames)
        
        # Convert video_unnorm back to numpy and dtype uint8
        video_unnorm = video_unnorm.permute(0, 2, 3, 1)
        video_unnorm = (video_unnorm * 255).to(torch.uint8)

        # Load mask
        object_masks, object_idx_masks = self.load_mask(clip_id, frame_indices)

        # Load audio
        waveform = self.load_audio(video_fp)
        
        # Extract fbank features of shape (frames, num_mel_bins)
        fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=self.args.audio_sample_rate, use_energy=False, window_type='hanning', num_mel_bins=self.args.num_mel_bins, dither=0.0, frame_shift=10)

        # normalize
        fbank = (fbank + 4.26) / (4.57 * 2)
        
        # Pad or cut if necessary
        n_frames = fbank.shape[0]

        p = self.args.ast_input_tdim - n_frames

        # cut and pad
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[0:self.args.ast_input_tdim, :]
   
        # cut or pad waveform
        if waveform.shape[1] < int(self.args.audio_sample_rate * 1.5):
            waveform = F.pad(waveform, (0, int(self.args.audio_sample_rate * 1.5) - waveform.shape[1]))
        elif waveform.shape[1] > int(self.args.audio_sample_rate * 1.5):
            waveform = waveform[:, :int(self.args.audio_sample_rate * 1.5)]
        
        return {'video': video_norm, 'video_np': video_unnorm, 'fbank': fbank, 'narration': text, 'is_sounding_action': is_sounding_action, 
                'action_group': action_group, 'clip_id': clip_id, 'object_masks': object_masks, 'object_idx_masks': object_idx_masks,
                'noun_vec': noun_vec, 'verb_vec': verb_vec, 'waveform': waveform}
    
    
    def get_noun_verb(self, sample):
        noun_vec = torch.zeros(self.noun_dim)
        verb_vec = torch.zeros(self.verb_dim)
        noun_idx = eval(sample['tag_noun'])
        verb_idx = eval(sample['tag_verb'])
        for i in noun_idx:
            noun_vec[i] = 1
        for i in verb_idx:
            verb_vec[i] = 1

        return noun_vec, verb_vec
    
    def load_mask(self, clip_id, frame_indices):
        mask_path = os.path.join(self.args.eval_mask_dir, f"{clip_id}.npz")
        loaded = np.load(mask_path, allow_pickle=True)
        mask_dict = load_npz_mask(loaded)

        # Get object masks for corresponding frame indices
        all_object_masks = mask_dict['object_masks']
        object_masks = [all_object_masks[int(frame)] for frame in frame_indices]
        object_masks = np.array(object_masks)
        object_masks = torch.tensor(object_masks)    
        object_masks = self.video_transform(object_masks)

        # Get mask indices for object masks
        all_object_idx_masks = mask_dict['object_idx_masks']
        object_idx_masks = [all_object_idx_masks[int(frame)] for frame in frame_indices]
        object_idx_masks = np.array(object_idx_masks)
        
        return object_masks, object_idx_masks
    
 
    def load_video(self, video_fp, num_frames):
        try:
            vr = VideoReader(video_fp.replace('.wav', '.mp4'), ctx=cpu(0))
            total_frames = len(vr)

            all_frame_indices = np.arange(total_frames)
            
            # We subsample the framerate by 3
            all_frame_indices = all_frame_indices[::3]

            # Uniformly sample num_frames frames
            uniform_samples = np.linspace(0, len(all_frame_indices) - 1, num_frames, dtype=int)
            frame_indices = all_frame_indices[uniform_samples]
            
            imgs = vr.get_batch(frame_indices)

        except Exception as e:
            print('failed to load video, use black image instead', e)
            imgs = torch.zeros(self.video_size)
            frame_indices = np.zeros(num_frames)

        imgs = (imgs / 255.0).permute(0, 3, 1, 2)  # [T, H, W, C] ---> [T, C, H, W]
        imgs = self.video_transform(imgs)
        imgs_norm = self.video_normalize(imgs)
        return imgs_norm, imgs, frame_indices

    def load_audio(self, audio_fp):
        try:
            ar = AudioReader(audio_fp.replace('.wav', '.mp4'), ctx=cpu(0), sample_rate=self.args.audio_sample_rate)
            waveform = ar[:]
            
        except Exception as e:
            print(f'Exception while reading audio file {audio_fp} with {e}')
            waveform = torch.zeros(self.waveform_size)
            
        return waveform


class Ego4D_Detection_Eval(Dataset):
    
    def __init__(self,
                args,
                ):
       
        self.args = args
        self.data_dir = os.path.expandvars(self.args.eval_clips_dir)  # check for environment variables

        metadata_file = self.args.eval_metadata_file

        self.metadata = pd.read_csv(metadata_file, sep='\t')

        # Remove rows where file does not exist
        # self.metadata = self.metadata[self.metadata['clip_file'].apply(lambda x: os.path.exists(os.path.join(self.data_dir, x).replace('.wav', '.mp4')))]

        self.seed = args.seed
        self.video_transform = transforms.Compose([
            transforms.Resize(args.video_input_res),
            transforms.CenterCrop(args.video_input_res),
        ])
        self.video_normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    
    def get_id(self, sample):
        if 'narration_source' in sample and 'narration_ind' in sample:
            return sample['video_uid'] + '_' + sample['narration_source'] + '_' + str(sample['narration_ind'])
        else:
            return sample['video_uid']

    def __len__(self):
        return len(self.metadata)

    @property
    def video_size(self):
        return (self.args.video_num_frames, self.args.video_input_res, self.args.video_input_res, 3)

    @property
    def spec_size(self):
        return (self.args.num_mel_bins, self.args.ast_input_tdim)

    @property
    def waveform_size(self):
        return (1, int(self.args.audio_sample_rate * self.args.audio_num_secs))
        
    
    def load_object_pool_masks(self, clip_id):
        self.num_pool_objects = 15
        mask_path = os.path.join(self.args.eval_pool_mask_dir, f"{clip_id}.pth")
        mask_dict = torch.load(mask_path, weights_only=False)
        object_pool_masks = mask_dict.mask

        # Check if the object pool masks are empty
        if len(object_pool_masks) == 0:
            # If empty, create a tensor of zeros with the same shape as the expected object pool masks
            object_pool_masks = np.zeros((1, 224, 224), dtype=bool)

        object_pool_masks = torch.tensor(object_pool_masks)
        object_pool_masks = self.video_transform(object_pool_masks)

        N, H, W = object_pool_masks.shape
        device = object_pool_masks.device
        # TODO: Each object pool has variable number of objects. We need to pad/truncate them to same size
        if N >= self.num_pool_objects:
            object_pool_masks = object_pool_masks[:self.num_pool_objects]
            object_pool_idx_masks = torch.ones(self.num_pool_objects, device=device)
        else:
            padding = torch.zeros((self.num_pool_objects - N, H, W), dtype=object_pool_masks.dtype, device=device)
            object_pool_masks = torch.cat([object_pool_masks, padding], dim=0)
            object_pool_idx_masks = torch.cat([
                torch.ones(N, device=device),
                torch.zeros(self.num_pool_objects - N, device=device)
            ])

        return object_pool_masks, object_pool_idx_masks

    
    def load_mask(self, clip_id, frame_indices):
        mask_path = os.path.join(self.args.eval_mask_dir, f"{clip_id}.npz")
        loaded = np.load(mask_path, allow_pickle=True)
        mask_dict = load_npz_mask(loaded)

        # Get object masks for corresponding frame indices    
        all_object_masks = mask_dict['object_masks']
        object_masks = [all_object_masks[min(frame, len(all_object_masks)-1)] for frame in frame_indices]
        object_masks = np.array(object_masks).squeeze()
        object_masks = torch.tensor(object_masks)
        object_masks = self.video_transform(object_masks)

        # Get mask indices for object masks
        all_object_idx_masks = mask_dict['object_idx_masks']
        object_idx_masks = [all_object_idx_masks[min(frame, len(all_object_idx_masks)-1)] for frame in frame_indices]
        object_idx_masks = np.array(object_idx_masks)

        is_sounding_action = mask_dict['is_sounding_action']

        object_labels = mask_dict['object_labels']

        if len(object_labels) == 0:
            object_labels.extend(['none'] * 2)
        
        # Check if None is in the object labels. If so, set it to 'none'
        if object_labels[0] is None:
            object_labels[0] = 'none'
        if object_labels[1] is None:
            object_labels[1] = 'none'

        # Check if 'hand' is in either of the object labels
        if 'hand' in object_labels[0]:
            # Set the first object mask to all False
            object_masks[0] = torch.zeros_like(object_masks[0], dtype=torch.bool)
        elif 'hand' in object_labels[1]:
            # Set the second object mask to all False
            object_masks[1] = torch.zeros_like(object_masks[1], dtype=torch.bool)
        
        return object_masks, object_idx_masks, is_sounding_action, object_labels

    
    def load_video(self, video_fp, num_frames):
        try:
            vr = VideoReader(video_fp, ctx=cpu(0))
            total_frames = len(vr)
            
            # Clips are 3 seconds long with the narration in the middle. We want to extract
            # 0.5 seconds before and 1 second after the narration.
            start_frame = int(total_frames / 3)
            end_frame = int(2.5 * total_frames / 3)  # 2.5s corresponds to 5/6 of the frames
          
            # Uniformly sample num_frames frames
            frame_indices = np.linspace(start_frame, end_frame, num_frames, dtype=int)

            imgs = vr.get_batch(frame_indices)

        except Exception as e:
            print('failed to load video, use black image instead', e)
            imgs = torch.zeros(self.video_size)
        
        imgs = imgs.permute(0, 3, 1, 2)  # [T, H, W, C] ---> [T, C, H, W]
        imgs = self.video_transform(imgs)
        imgs = imgs / 255.0
        imgs_norm = self.video_normalize(imgs)

        return imgs_norm, imgs, frame_indices

    def load_audio(self, audio_fp):
        try:
            ar = AudioReader(audio_fp.replace('.wav', '.mp4'), ctx=cpu(0), sample_rate=self.args.audio_sample_rate)
            waveform = ar[:]
            
            start_idx = int(self.args.audio_sample_rate * 1)  # start at 1s
            end_idx = int(self.args.audio_sample_rate * 2.5)  # end at 2.5s
            waveform = waveform[:, start_idx:end_idx]
        except Exception as e:
            print(f'Exception while reading audio file {audio_fp} with {e}')
            waveform = torch.zeros(self.waveform_size)
            
        return waveform


    def __getitem__(self, item):
        item = item % len(self.metadata)
        sample = self.metadata.iloc[item]
        narration_time = math.ceil(sample["narration_time"])
        clip_id = os.path.basename(sample['clip_file'])[:-4]
        video_fp = os.path.join(self.data_dir, f"{clip_id}.mp4")
        text = sample['clip_text']
        if 'action_group' in sample:
            action_group = sample['action_group']
        else:
            action_group = -1

        # Load video
        video_norm, video_unnorm, frame_indices = self.load_video(video_fp, num_frames=self.args.video_num_frames)
        
        # Convert video_unnorm back to numpy and dtype uint8
        video_unnorm = video_unnorm.permute(0, 2, 3, 1)
        video_unnorm = (video_unnorm * 255).to(torch.uint8)

        object_masks, object_idx_masks, is_sounding_action, object_labels = self.load_mask(clip_id, frame_indices)

        object_pool_masks, object_pool_idx_masks = self.load_object_pool_masks(clip_id)

        # Load audio
        waveform = self.load_audio(video_fp)
        
        # Extract fbank features of shape (frames, num_mel_bins)
        fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=self.args.audio_sample_rate, use_energy=False, window_type='hanning', num_mel_bins=self.args.num_mel_bins, dither=0.0, frame_shift=10)
        
        # normalize
        fbank = (fbank + 4.26) / (4.57 * 2)
        
        # Pad or cut if necessary
        n_frames = fbank.shape[0]

        p = self.args.ast_input_tdim - n_frames

        # cut and pad
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[0:self.args.ast_input_tdim, :]
                
        return {'video': video_norm, 'video_np': video_unnorm, 'fbank': fbank, 'wav': waveform, 'narration': text, 'clip_id': clip_id, 'object_masks': object_masks, 'object_pool_masks': object_pool_masks, 'object_idx_masks': object_idx_masks, 
                'object_pool_idx_masks': object_pool_idx_masks, 'is_sounding_action': is_sounding_action, 'action_group': action_group, 'object_labels': object_labels}


class EpicKitchensDataset(Dataset):
    """Dataset class for handling video and audio data from the Epic Kitchens dataset.
    This class supports loading and preprocessing of video and audio clips based on metadata.
    """

    def __init__(self,
                args,
                ):
       
        self.args = args
        self.data_dir = os.path.expandvars(self.args.clips_dir)  # check for environment variables

        metadata_file = self.args.train_metadata_file

        self.metadata = pd.read_csv(metadata_file, on_bad_lines='warn')

        # Remove rows where file does not exist
        # self.metadata = self.metadata[self.metadata['clip_file'].apply(lambda x: os.path.exists(os.path.join(self.data_dir, x).replace('.wav', '.mp4')))]

        self.seed = args.seed
        self.video_transform = transforms.Compose([
            transforms.Resize(args.video_input_res),
            transforms.CenterCrop(args.video_input_res),
        ])
        self.video_normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.metadata)

    @property
    def video_size(self):
        return (self.args.video_num_frames, self.args.video_input_res, self.args.video_input_res, 3)

    @property
    def spec_size(self):
        return (self.args.num_mel_bins, self.args.ast_input_tdim)

    @property
    def waveform_size(self):
        return (1, int(self.args.audio_sample_rate * self.args.audio_num_secs))

    def __getitem__(self, item):

        item = item % len(self.metadata)
        sample = self.metadata.iloc[item]
        clip_id = sample['narration_id']
        video_fp = os.path.join(self.data_dir, f"{clip_id}.mp4")
        # logging.info(f"loading video {video_fp}")
        text = sample['narration']

        # Load video
        video_norm, video_unnorm, frame_indices = self.load_video(video_fp, num_frames=self.args.video_num_frames)
        
        # Convert video_unnorm back to numpy and dtype uint8
        video_unnorm = video_unnorm.permute(0, 2, 3, 1)
        video_unnorm = (video_unnorm * 255).to(torch.uint8)
     
        # Load audio
        waveform = self.load_audio(video_fp, frame_indices)
        
        # Extract fbank features of shape (frames, num_mel_bins)
        fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=self.args.audio_sample_rate, use_energy=False, window_type='hanning', num_mel_bins=self.args.num_mel_bins, dither=0.0, frame_shift=10)
        
        # normalize
        fbank = (fbank + 4.26) / (4.57 * 2)
        
        # Pad or cut if necessary
        n_frames = fbank.shape[0]

        p = self.args.ast_input_tdim - n_frames

        # cut and pad
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[0:self.args.ast_input_tdim, :]
   
        # Load mask
        object_masks, object_idx_masks = self.load_mask(clip_id, frame_indices)  # (num_frames, num_objects, H, W), (num_frames, num_objects)
        
        return {'video': video_norm, 'video_np': video_unnorm, 'wav': waveform, 'fbank': fbank, 'narration': text, 'clip_id': clip_id, 'object_masks': object_masks, 'object_idx_masks': object_idx_masks}
    
    
    def load_mask(self, clip_id, frame_indices):
        mask_path = os.path.join(self.args.mask_dir, f"{clip_id}.pth")
        mask_dict = torch.load(mask_path, weights_only=False)

        # Get object masks for corresponding frame indices    
        all_object_masks = mask_dict['object_masks']
        object_masks = [all_object_masks[min(frame, len(all_object_masks)-1)] for frame in frame_indices]
        object_masks = np.array(object_masks).squeeze(2)
        object_masks = torch.tensor(object_masks)    
        object_masks = self.video_transform(object_masks)

        # Get mask indices for object masks
        all_object_idx_masks = mask_dict['object_idx_masks']
        object_idx_masks = [all_object_idx_masks[min(frame, len(all_object_idx_masks)-1)] for frame in frame_indices]
        object_idx_masks = np.array(object_idx_masks)
        
        return object_masks, object_idx_masks

 
    def load_video(self, video_fp, num_frames):
        try:
            vr = VideoReader(video_fp, ctx=cpu(0))
            total_frames = len(vr)
            
            # Clips are 3 seconds long with the narration in the middle. We want to extract
            # 0.5 seconds before and 1 second after the narration.
            start_frame = int(total_frames / 3)
            end_frame = int(2.5 * total_frames / 3)  # 2.5s corresponds to 5/6 of the frames
            all_frames = np.arange(start_frame, end_frame)
            
            # Randomly sample num_frames frames
            frame_indices = np.random.choice(all_frames, num_frames, replace=False)
            frame_indices.sort()

            imgs = vr.get_batch(frame_indices)

        except Exception as e:
            print('failed to load video, use black image instead', e)
            imgs = torch.zeros(self.video_size)
        
        imgs = imgs.permute(0, 3, 1, 2)  # [T, H, W, C] ---> [T, C, H, W]
        imgs = self.video_transform(imgs)
        imgs = imgs / 255.0
        imgs_norm = self.video_normalize(imgs)

        return imgs_norm, imgs, frame_indices

    
    def load_audio(self, audio_fp, frame_indices):
        try:
            ar = AudioReader(audio_fp.replace('.wav', '.mp4'), ctx=cpu(0), sample_rate=self.args.audio_sample_rate)
            waveform = ar[:]
            
            start_idx = int(self.args.audio_sample_rate * 1)  # start at 1s
            end_idx = int(self.args.audio_sample_rate * 2.5)  # end at 2.5s
            waveform = waveform[:, start_idx:end_idx]
        
        except Exception as e:
            print(f'Exception while reading audio file {audio_fp} with {e}')
            waveform = torch.zeros(self.waveform_size)
            
        return waveform


class EpicKitchens_Detection_Eval(EpicKitchensDataset):
    
    def __init__(self,
                args,
                ):
       
        self.args = args
        self.data_dir = os.path.expandvars(self.args.eval_clips_dir)  # check for environment variables

        metadata_file = self.args.eval_metadata_file

        self.metadata = pd.read_csv(metadata_file, on_bad_lines='warn')

        # Remove rows where file does not exist
        # self.metadata = self.metadata[self.metadata['clip_file'].apply(lambda x: os.path.exists(os.path.join(self.data_dir, x).replace('.wav', '.mp4')))]

        self.seed = args.seed
        self.video_transform = transforms.Compose([
            transforms.Resize(args.video_input_res),
            transforms.CenterCrop(args.video_input_res),
        ])
        self.video_normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    
    def load_mask(self, clip_id, frame_indices):
        mask_path = os.path.join(self.args.eval_mask_dir, f"{clip_id}.pth")
        mask_dict = torch.load(mask_path, weights_only=False)

        # Get object masks for corresponding frame indices    
        all_object_masks = mask_dict['object_masks']
        object_masks = [all_object_masks[min(frame, len(all_object_masks)-1)] for frame in frame_indices]
        object_masks = np.array(object_masks).squeeze()
        object_masks = torch.tensor(object_masks)    
        object_masks = self.video_transform(object_masks)

        # Get mask indices for object masks
        all_object_idx_masks = mask_dict['object_idx_masks']
        object_idx_masks = [all_object_idx_masks[min(frame, len(all_object_idx_masks)-1)] for frame in frame_indices]
        object_idx_masks = np.array(object_idx_masks)

        is_sounding_action = mask_dict['is_sounding_action']

        object_labels = mask_dict['object_labels']

        if len(object_labels) == 0:
            object_labels.extend(['none'] * 2)

        # Check if 'hand' is in either of the object labels
        if 'hand' in object_labels[0]:
            # Set the first object mask to all False
            object_masks[0] = torch.zeros_like(object_masks[0], dtype=torch.bool)
        elif 'hand' in object_labels[1]:
            # Set the second object mask to all False
            object_masks[1] = torch.zeros_like(object_masks[1], dtype=torch.bool)
        
        return object_masks, object_idx_masks, is_sounding_action, object_labels

    
    def load_object_pool_masks(self, clip_id):
        self.num_pool_objects = 15
        mask_path = os.path.join(self.args.eval_pool_mask_dir, f"{clip_id}.pth")
        mask_dict = torch.load(mask_path, weights_only=False)
        object_pool_masks = mask_dict.mask

        # Check if the object pool masks are empty
        if len(object_pool_masks) == 0:
            # If empty, create a tensor of zeros with the same shape as the expected object pool masks
            object_pool_masks = np.zeros((1, 224, 224), dtype=bool)

        object_pool_masks = torch.tensor(object_pool_masks)
        object_pool_masks = self.video_transform(object_pool_masks)

        N, H, W = object_pool_masks.shape
        device = object_pool_masks.device
        # TODO: Each object pool has variable number of objects. We need to pad/truncate them to same size
        if N >= self.num_pool_objects:
            object_pool_masks = object_pool_masks[:self.num_pool_objects]
            object_pool_idx_masks = torch.ones(self.num_pool_objects, device=device)
        else:
            padding = torch.zeros((self.num_pool_objects - N, H, W), dtype=object_pool_masks.dtype, device=device)
            object_pool_masks = torch.cat([object_pool_masks, padding], dim=0)
            object_pool_idx_masks = torch.cat([
                torch.ones(N, device=device),
                torch.zeros(self.num_pool_objects - N, device=device)
            ])

        return object_pool_masks, object_pool_idx_masks
    
    
    def load_video(self, video_fp, num_frames):
        try:
            vr = VideoReader(video_fp, ctx=cpu(0))
            total_frames = len(vr)
            # Clips are 3 seconds long with the narration in the middle. We want to extract
            # 0.5 seconds before and 1 second after the narration.
            start_frame = int(total_frames / 3)
            end_frame = int(2.5 * total_frames / 3)  # 2.5s corresponds to 5/6 of the frames
          
            # Uniformly sample num_frames frames
            frame_indices = np.linspace(start_frame, end_frame, num_frames, dtype=int)

            imgs = vr.get_batch(frame_indices)

        except Exception as e:
            print('failed to load video, use black image instead', e)
            imgs = torch.zeros(self.video_size)
        
        imgs = imgs.permute(0, 3, 1, 2)  # [T, H, W, C] ---> [T, C, H, W]
        imgs = self.video_transform(imgs)
        imgs = imgs / 255.0
        imgs_norm = self.video_normalize(imgs)

        return imgs_norm, imgs, frame_indices


    def __getitem__(self, item):

        item = item % len(self.metadata)
        sample = self.metadata.iloc[item]
        clip_id = sample['narration_id']
        video_fp = os.path.join(self.data_dir, f"{clip_id}.mp4")
        # logging.info(f"loading video {video_fp}")
        text = sample['narration']
        action_group = sample['action_group_index']

        # Load video
        video_norm, video_unnorm, frame_indices = self.load_video(video_fp, num_frames=self.args.video_num_frames)
        
        # Convert video_unnorm back to numpy and dtype uint8
        video_unnorm = video_unnorm.permute(0, 2, 3, 1)
        video_unnorm = (video_unnorm * 255).to(torch.uint8)

        object_masks, object_idx_masks, is_sounding_action, object_labels = self.load_mask(clip_id, frame_indices)

        object_pool_masks, object_pool_idx_masks = self.load_object_pool_masks(clip_id)

        # Load audio
        waveform = self.load_audio(video_fp, frame_indices)
        
        # Extract fbank features of shape (frames, num_mel_bins)
        fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=self.args.audio_sample_rate, use_energy=False, window_type='hanning', num_mel_bins=self.args.num_mel_bins, dither=0.0, frame_shift=10)
        
        # normalize
        fbank = (fbank + 4.26) / (4.57 * 2)
        
        # Pad or cut if necessary
        n_frames = fbank.shape[0]

        p = self.args.ast_input_tdim - n_frames

        # cut and pad
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[0:self.args.ast_input_tdim, :]
   
        return {'video': video_norm, 'video_np': video_unnorm, 'wav': waveform, 'fbank': fbank, 'narration': text, 'clip_id': clip_id, 'object_masks': object_masks, 'object_idx_masks': object_idx_masks, 
                'object_pool_masks': object_pool_masks, 'object_pool_idx_masks': object_pool_idx_masks, 'is_sounding_action': is_sounding_action, 'action_group': action_group, 'object_labels': object_labels}
