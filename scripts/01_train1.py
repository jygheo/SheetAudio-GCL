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
from src.models import SheetMusicTeacherGAT, SpectrogramSwin, SymmetricCrossModalMoCo
from src.data import MSMDDataset, CLASS_VOCAB, get_deterministic_splits, custom_collate_fn
from src.evaluate import evaluate_graph_audio_retrieval

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    save_dir = './checkpoints'
    os.makedirs(save_dir, exist_ok=True)
    dataset_root = './data/msmd_dataset/msmd_aug_v1-1_no-audio'
    
    train_pieces, val_pieces, _ = get_deterministic_splits(dataset_root)
    train_loader = DataLoader(MSMDDataset(dataset_root, train_pieces, CLASS_VOCAB), batch_size=32, shuffle=True, collate_fn=custom_collate_fn, num_workers=2, drop_last=True)
    val_loader = DataLoader(MSMDDataset(dataset_root, val_pieces, CLASS_VOCAB, mode='val'), batch_size=32, shuffle=False, collate_fn=custom_collate_fn, num_workers=2)

    graph_encoder = SheetMusicTeacherGAT(num_classes=len(CLASS_VOCAB)).to(device)
    audio_encoder = SpectrogramSwin(out_channels=512).to(device)
    moco_model = SymmetricCrossModalMoCo(graph_encoder, audio_encoder, K=16384).to(device)

    optimizer = optim.AdamW(moco_model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = SequentialLR(optimizer, schedulers=[
        LinearLR(optimizer, start_factor=1e-6, end_factor=1.0, total_iters=1000),
        CosineAnnealingLR(optimizer, T_max=19)
    ], milestones=[1])

    best_loss = float('inf')
    
    for epoch in range(20):
        moco_model.train()
        total_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/20 Phase 1")
        
        for batch in progress:
            audio_inputs = batch['spectrograms'].to(device)
            freq_mask, time_mask = T_audio.FrequencyMasking(15), T_audio.TimeMasking(30)
            audio_inputs = torch.stack([time_mask(freq_mask(x)) for x in audio_inputs.unbind(0)])
            
            graph_inputs = {
                'x_cont': batch['graph_x_cont'].to(device), 'x_class': batch['graph_x_class'].to(device),
                'x_pitch': batch['graph_x_pitch'].to(device), 'edge_index': batch['graph_edge_index'].to(device),
                'batch': batch['graph_batch_index'].to(device)
            }
            graph_inputs['x_cont'][:, :2] = torch.clamp(graph_inputs['x_cont'][:, :2] + (torch.rand_like(graph_inputs['x_cont'][:, :2]) - 0.5) * 0.02, 0.0, 1.0)
            
            optimizer.zero_grad()
            loss = moco_model(graph_inputs, audio_inputs)
            loss.backward()
            clip_grad_norm_(moco_model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            progress.set_postfix({'Loss': f"{loss.item():.4f}"})
            
        avg_loss = total_loss / len(train_loader)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({'model_state_dict': moco_model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, os.path.join(save_dir, "phase1_moco_best.pth"))
            
        metrics = evaluate_graph_audio_retrieval(moco_model, val_loader, device)
        print(f"Phase 1 A2S R@1: {metrics['A2S']['R@1']:.3f} | S2A R@1: {metrics['S2A']['R@1']:.3f}")

if __name__ == '__main__':
    main()