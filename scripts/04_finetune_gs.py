import os
import sys
import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
import torchaudio.transforms as T_audio
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models import SheetMusicSwin, SpectrogramSwin, VisionAudioMoCo
from src.data import GrandstaffFinetuneDataset, eval_collate_fn
from src.transforms import MSMDSpectrogramPipeline, get_vision_augmenter
from src.evaluate import evaluate_vision_audio_retrieval

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    save_dir = './checkpoints'
    
    # Download HuggingFace components for initialization
    audio_path = hf_hub_download(repo_id="hyg444/SheetAudio-GCL", filename="SheetAudio-GCL-Audio.pth")
    vision_path = hf_hub_download(repo_id="hyg444/SheetAudio-GCL", filename="SheetAudio-GCL-Sheet.pth")
    filterbank_path = hf_hub_download(repo_id="hyg444/SheetAudio-GCL", filename="madmom_filterbank.pt")

    vision_encoder = SheetMusicSwin().to(device)
    audio_encoder = SpectrogramSwin().to(device)
    vision_encoder.load_state_dict(torch.load(vision_path, map_location=device))
    audio_encoder.load_state_dict(torch.load(audio_path, map_location=device))

    audio_pipeline = MSMDSpectrogramPipeline(torch.load(filterbank_path, map_location='cpu'))
    audio_pipeline.eval()

    hf_train = load_dataset('parquet', data_files={'train': 'hf://datasets/PRAIG/grandstaff-grandstaff-multimodal/data/train-*.parquet'}, split='train')
    hf_val = load_dataset('parquet', data_files={'val': 'hf://datasets/PRAIG/grandstaff-grandstaff-multimodal/data/val-*.parquet'}, split='val')

    train_loader = DataLoader(GrandstaffFinetuneDataset(hf_train, audio_pipeline, mode='train'), batch_size=32, shuffle=True, collate_fn=eval_collate_fn, num_workers=2, drop_last=True)
    val_loader = DataLoader(GrandstaffFinetuneDataset(hf_val, audio_pipeline, mode='val'), batch_size=32, shuffle=False, collate_fn=eval_collate_fn, num_workers=2)

    moco_model = VisionAudioMoCo(vision_encoder, audio_encoder, K=16384).to(device)
    optimizer = optim.AdamW(moco_model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = SequentialLR(optimizer, schedulers=[LinearLR(optimizer, start_factor=1e-6, end_factor=1.0, total_iters=1000), CosineAnnealingLR(optimizer, T_max=14)], milestones=[1])
    vision_augmenter = get_vision_augmenter()

    best_loss = float('inf')
    for epoch in range(15):
        moco_model.train()
        total_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/15 Grandstaff Finetune")
        
        for batch in progress:
            images = vision_augmenter(batch['images'].to(device))
            audio_inputs = batch['spectrograms'].to(device)
            freq_mask, time_mask = T_audio.FrequencyMasking(15), T_audio.TimeMasking(30)
            audio_inputs = torch.stack([time_mask(freq_mask(x)) for x in audio_inputs.unbind(0)])
            
            optimizer.zero_grad()
            loss = moco_model(images, audio_inputs)
            loss.backward()
            clip_grad_norm_(moco_model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            progress.set_postfix({'Loss': f"{loss.item():.4f}"})

        avg_loss = total_loss / len(train_loader)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({'model_state_dict': moco_model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, os.path.join(save_dir, "grandstaff_finetune_best.pth"))
            
        metrics = evaluate_vision_audio_retrieval(moco_model.encoder_q_vision, moco_model.encoder_q_audio, val_loader, device)
        print(f"Finetune A2V R@1: {metrics['A2V']['R@1']:.3f} | V2A R@1: {metrics['V2A']['R@1']:.3f}")

if __name__ == '__main__':
    main()