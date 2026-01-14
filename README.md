# clink-chop-thud

Sounding Object Detection Train: `torchrun train_detection.py --checkpoint /coc/flash5/myang415/objects_project_clean/infonce_freezeslots.pth --dataset epic_kitchens`

Sounding Object Detection Eval: `torchrun train_detection.py --checkpoint /coc/flash5/myang415/objects_project_clean/objects_project_localization/355520/objects_model_latest.pth --eval --dataset epic_kitchens`

Sounding Action Discovery Train: `torchrun train.py`

Sounding Action Discovery Eval: `torchrun train.py --eval --eval_metadata_file /coc/flash5/myang415/objects_project_clean/metadata_files/ego4dsounds_eval_masks.csv --checkpoint /coc/flash5/myang415/objects_project_clean/infonce_freezeslots.pth`