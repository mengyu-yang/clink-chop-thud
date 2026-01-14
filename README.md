# Clink! Chop! Thud! — Learning Object Sounds from Real-World Interactions

[Project Page](https://clink-chop-thud.github.io/) | [arXiv](https://arxiv.org/abs/2510.02313)


## Motivation

<p align="center">
  <img src="/coc/flash5/myang415/clink-chop-thud/figures/teaser_fig.png" alt="Teaser Figure" width="80%">
</p>

Humans handle a wide variety of objects throughout the day and many of these interactions produce sounds. We introduce a multimodal *object-aware* framework that learns the relationship between the objects in an interaction and the resulting sounds. This enables our model to detect the *sounding objects* from a set of candidates in a scene.

## Data

## Pretrained Models



Sounding Object Detection Train: `torchrun train_detection.py --checkpoint /coc/flash5/myang415/objects_project_clean/infonce_freezeslots.pth --dataset epic_kitchens`

Sounding Object Detection Eval: `torchrun train_detection.py --checkpoint /coc/flash5/myang415/objects_project_clean/objects_project_localization/355520/objects_model_latest.pth --eval --dataset epic_kitchens`

Sounding Action Discovery Train: `torchrun train.py`

Sounding Action Discovery Eval: `torchrun train.py --eval --eval_metadata_file /coc/flash5/myang415/objects_project_clean/metadata_files/ego4dsounds_eval_masks.csv --checkpoint /coc/flash5/myang415/objects_project_clean/infonce_freezeslots.pth`

## Bibtex

```bibtex
@inproceedings{yang2025clink,
    title = {Clink! Chop! Thud! -- Learning Object Sounds from Real-World Interactions},
    author = {Mengyu Yang and Yiming Chen and Haozheng Pei and Siddhant Agarwal and Arun Balajee Vasudevan and James Hays},
    year = {2025},
    booktitle = {ICCV},
}