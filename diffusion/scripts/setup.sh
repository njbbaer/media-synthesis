#!/bin/bash

# Disable tmux
touch ~/.no_auto_tmux

# Install packages
apt install git curl -y
pip install ftfy regex requests tqdm jupyterlab

# Clone repository
git clone --recursive https://github.com/crowsonkb/v-diffusion-pytorch

# Download diffusion model
mkdir v-diffusion-pytorch/checkpoints
curl -L https://v-diffusion.s3.us-west-2.amazonaws.com/cc12m_1_cfg.pth > v-diffusion-pytorch/checkpoints/cc12m_1_cfg.pth