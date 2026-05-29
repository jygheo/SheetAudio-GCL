#!/bin/bash
# Install aria2 and p7zip if not already installed (Debian/Ubuntu)
# sudo apt-get update && sudo apt-get install -y aria2 p7zip-full

echo "Downloading MSMD dataset..."
aria2c -x 16 -s 16 -k 1M -q -o msmd.zip.part \
"https://zenodo.org/record/2597505/files/msmd_aug_v1-1_no-audio.zip"

echo "Extracting MSMD dataset..."
mkdir -p ./data/msmd_dataset
7z x msmd.zip.part -o./data/msmd_dataset -mmt=on

mkdir -p ./checkpoints

echo "Setup complete."