import os
import sys
import torch
from torch.utils.data import DataLoader
from huggingface_hub import hf_hub_download

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models import SheetMusicSwin, SpectrogramSwin
from src.data import MSMDNoGraphDataset, get_deterministic_splits, eval_collate_fn, set_seed
from src.evaluate import evaluate_vision_audio_retrieval

def main():
    set_seed(42)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    dataset_root = './data/msmd_dataset/msmd_aug_v1-1_no-audio'
    if not os.path.exists(dataset_root):
        print(f"MSMD root '{dataset_root}' not found. Please run setup.sh to download it.")
        return

    print("Downloading HuggingFace weights for evaluation...")
    audio_path = hf_hub_download(repo_id="hyg444/SheetAudio-GCL", filename="SheetAudio-GCL-Audio.pth")
    vision_path = hf_hub_download(repo_id="hyg444/SheetAudio-GCL", filename="SheetAudio-GCL-Sheet.pth")

    print("Initializing models...")
    vision_encoder = SheetMusicSwin().to(device)
    audio_encoder = SpectrogramSwin().to(device)

    vision_encoder.load_state_dict(torch.load(vision_path, map_location=device))
    audio_encoder.load_state_dict(torch.load(audio_path, map_location=device))

    vision_encoder.eval()
    audio_encoder.eval()
    _, _, test_pieces = get_deterministic_splits(dataset_root)

    print("Building MSMD Test Dataset...")
    msmd_test_dataset = MSMDNoGraphDataset(root_dir=dataset_root, split_pieces=test_pieces)
    
    msmd_test_loader = DataLoader(
        msmd_test_dataset,
        batch_size=64,
        shuffle=False, 
        collate_fn=eval_collate_fn,
        num_workers=2
    )

    print("\n--- Evaluating on MSMD Test Split ---")
    msmd_metrics = evaluate_vision_audio_retrieval(vision_encoder, audio_encoder, msmd_test_loader, device)
    
    print("Audio-to-Vision:")
    print(f"  R@1: {msmd_metrics['A2V']['R@1']:.3f} | R@25: {msmd_metrics['A2V']['R@25']:.3f} | MRR: {msmd_metrics['A2V']['MRR']:.3f} | MR: {msmd_metrics['A2V']['MR']}")
    print("Vision-to-Audio:")
    print(f"  R@1: {msmd_metrics['V2A']['R@1']:.3f} | R@25: {msmd_metrics['V2A']['R@25']:.3f} | MRR: {msmd_metrics['V2A']['MRR']:.3f} | MR: {msmd_metrics['V2A']['MR']}")

if __name__ == '__main__':
    main()