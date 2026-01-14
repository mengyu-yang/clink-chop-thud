model_args = {
    "image_size": 224,
    "val_image_size": 224,
    "val_mask_size": 320,
    "eval_batch_size": 32,
    "viz_resolution_factor": 0.5,

    "checkpoint_path": "/coc/flash5/myang415/objects_project_clean/spot/spot_coco_checkpoint.pt.tar",
    "log_path": "results",
    "dataset": "coco",
    "data_path": None,  # Requires manual input

    "num_dec_blocks": 4,
    "d_model": 768,
    "num_heads": 6,
    "dropout": 0.0,

    "num_iterations": 3,
    "num_slots": 7,
    # "num_slots": 6,
    "slot_size": 256,
    "mlp_hidden_size": 1024,
    "img_channels": 3,
    "pos_channels": 4,
    "num_cross_heads": 6,

    "dec_type": "transformer",
    "cappa": -1,
    "mlp_dec_hidden": 2048,
    "use_slot_proj": True,

    "which_encoder": "dino_vitb16",
    "finetune_blocks_after": 100,
    "encoder_final_norm": False,

    "truncate": "bi-level",
    "init_method": "embedding",

    "use_second_encoder": True,

    "train_permutations": "random",
    "eval_permutations": "all",
}