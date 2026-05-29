import os
import sys
import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
import torchaudio.transforms as T_audio
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models import SheetMusicTeacherGAT, SpectrogramSwin, SheetMusicSwin, SymmetricCrossModalMoCo, VisionAudioMoCo
from src.data import MSMDDataset, CLASS_VOCAB, get_deterministic_splits, custom_collate_fn
from src.transforms import get_vision_augmenter
from src.evaluate import evaluate_vision_audio_retrieval

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    save_dir = './checkpoints'
    dataset_root = './data/msmd_dataset/msmd_aug_v1-1_no-audio'
    
    train_pieces, val_pieces, _ = get_deterministic_splits(dataset_root)
    train_loader = DataLoader(MSMDDataset(dataset_root, train_pieces, CLASS_VOCAB), batch_size=2, shuffle=True, collate_fn=custom_collate_fn, num_workers=2, drop_last=True)
    val_loader = DataLoader(MSMDDataset(dataset_root, val_pieces, CLASS_VOCAB, mode='val'), batch_size=2, shuffle=False, collate_fn=custom_collate_fn, num_workers=2)

    teacher_moco = SymmetricCrossModalMoCo(SheetMusicTeacherGAT(num_classes=len(CLASS_VOCAB)), SpectrogramSwin(), K=16384).to(device)
    teacher_moco.load_state_dict(torch.load(os.path.join(save_dir, "phase1_moco_best.pth"), map_location=device)['model_state_dict'])
    
    vision_student = SheetMusicSwin(out_channels=512).to(device)
    vision_student.load_state_dict(torch.load(os.path.join(save_dir, "phase2_vision_student_best.pth"), map_location=device)['model_state_dict'])

    moco_model = VisionAudioMoCo(vision_encoder=vision_student, audio_encoder=teacher_moco.encoder_q_audio, K=16384).to(device)

    optimizer = optim.AdamW(moco_model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = SequentialLR(optimizer, schedulers=[LinearLR(optimizer, start_factor=1e-6, end_factor=1.0, total_iters=1000), CosineAnnealingLR(optimizer, T_max=29)], milestones=[1])
    vision_augmenter = get_vision_augmenter()
    best_loss = float('inf')

    for epoch in range(30):
        moco_model.train()
        total_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/30 Phase 3")
        
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
            torch.save({'model_state_dict': moco_model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, os.path.join(save_dir, "phase3_moco_best.pth"))
            
        metrics = evaluate_vision_audio_retrieval(moco_model.encoder_q_vision, moco_model.encoder_q_audio, val_loader, device)
        print(f"Phase 3 A2V R@1: {metrics['A2V']['R@1']:.3f} | V2A R@1: {metrics['V2A']['R@1']:.3f}")

if __name__ == '__main__':
    main()