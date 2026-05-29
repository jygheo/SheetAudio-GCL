import os
import sys
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from huggingface_hub import hf_hub_download

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models import SheetMusicSwin, SpectrogramSwin, VisionAudioMoCo
from src.data import GrandstaffFinetuneDataset, eval_collate_fn, set_seed
from src.transforms import MSMDSpectrogramPipeline
from src.evaluate import evaluate_vision_audio_retrieval


def main():
    set_seed(42) 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint_path = './checkpoints/grandstaff_finetune_best.pth'

    filterbank_path = hf_hub_download(repo_id="hyg444/SheetAudio-GCL", filename="madmom_filterbank.pt")
    audio_pipeline = MSMDSpectrogramPipeline(torch.load(filterbank_path, map_location='cpu')).eval()

    moco_model = VisionAudioMoCo(SheetMusicSwin(), SpectrogramSwin(), K=16384).to(device)
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint {checkpoint_path}...")
        moco_model.load_state_dict(torch.load(checkpoint_path, map_location=device)['model_state_dict'])
    else:
        print("Checkpoint not found! Run finetuning scripts first.")
        return

    finetuned_vision = moco_model.encoder_q_vision.eval()
    finetuned_audio = moco_model.encoder_q_audio.eval()

    print("Evaluating on PRAIG/grandstaff-grandstaff-multimodal (Test Split)")
    hf_test = load_dataset('parquet', data_files={'test': 'hf://datasets/PRAIG/grandstaff-grandstaff-multimodal/data/test-*.parquet'}, split='test')
    
    test_loader = DataLoader(GrandstaffFinetuneDataset(hf_test, audio_pipeline, mode='val'), batch_size=32, shuffle=False, collate_fn=eval_collate_fn, num_workers=2)

    metrics = evaluate_vision_audio_retrieval(finetuned_vision, finetuned_audio, test_loader, device)

    print("\n--- Final Test Results on Grandstaff ---")
    print("Audio-to-Vision:")
    print(f"  R@1: {metrics['A2V']['R@1']:.3f} | R@25: {metrics['A2V']['R@25']:.3f} | MRR: {metrics['A2V']['MRR']:.3f} | MR: {metrics['A2V']['MR']}")
    print("Vision-to-Audio:")
    print(f"  R@1: {metrics['V2A']['R@1']:.3f} | R@25: {metrics['V2A']['R@25']:.3f} | MRR: {metrics['V2A']['MRR']:.3f} | MR: {metrics['V2A']['MR']}")

if __name__ == '__main__':
    main()