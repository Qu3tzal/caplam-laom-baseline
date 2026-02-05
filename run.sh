#!/bin/bash

# source .venv/bin/activate
python -m train_laom_labels --dataset.repo_id="HuggingFaceVLA/libero" --dataset.target_img_size=64
