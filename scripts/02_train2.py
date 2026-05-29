import os
import sys
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models import SheetMusicTeacherGAT, SpectrogramSwin, SheetMusicSwin, SymmetricCrossModalMoCo
from src.data import MSMDDataset, CLASS_VOCAB, get_deterministic_splits, custom_collate_fn
from src.transforms import get_vision_augmenter
from src.evaluate import evaluate_vision_audio_retrieval

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    save_dir = './checkpoints'
    dataset_root = './data/msmd_dataset/msmd_aug_v1-1_no-audio'
    
    train_pieces, val_pieces, _ = get_deterministic_splits(dataset_root)
    train_loader = DataLoader(MSMDDataset(dataset_root, train_pieces, CLASS_VOCAB), batch_size=32, shuffle=True, collate_fn=custom_collate_fn, num_workers=2, drop_last=True)
    val_loader = DataLoader(MSMDDataset(dataset_root, val_pieces, CLASS_VOCAB, mode='val'), batch_size=32, shuffle=False, collate_fn=custom_collate_fn, num_workers=2)

    # Load frozen Phase 1 teachers
    teacher_moco = SymmetricCrossModalMoCo(SheetMusicTeacherGAT(num_classes=len(CLASS_VOCAB)), SpectrogramSwin(), K=16384).to(device)
    teacher_moco.load_state_dict(torch.load(os.path.join(save_dir, "phase1_moco_best.pth"), map_location=device)['model_state_dict'])
    graph_teacher, audio_teacher = teacher_moco.encoder_q_graph.eval(), teacher_moco.encoder_q_audio.eval()
    for p in graph_teacher.parameters(): p.requires_grad = False
    for p in audio_teacher.parameters(): p.requires_grad = False

    vision_student = SheetMusicSwin(out_channels=512).to(device)
    optimizer = optim.AdamW(vision_student.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = SequentialLR(optimizer, schedulers=[
        LinearLR(optimizer, start_factor=1e-6, end_factor=1.0, total_iters=1000),
        CosineAnnealingLR(optimizer, T_max=29)
    ], milestones=[1])

    vision_augmenter = get_vision_augmenter()
    best_loss = float('inf')

    for epoch in range(30):
        vision_student.train()
        total_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/30 Phase 2")
        
        for batch in progress:
            images = vision_augmenter(batch['images'].to(device))
            graph_inputs = {
                'x_cont': batch['graph_x_cont'].to(device), 'x_class': batch['graph_x_class'].to(device),
                'x_pitch': batch['graph_x_pitch'].to(device), 'edge_index': batch['graph_edge_index'].to(device),
                'batch': batch['graph_batch_index'].to(device)
            }
            
            optimizer.zero_grad()
            with torch.no_grad(): target_g_embed = graph_teacher(**graph_inputs)
            v_embed = vision_student(images)
            
            loss = 1.0 - F.cosine_similarity(v_embed, target_g_embed, dim=-1).mean()
            loss.backward()
            clip_grad_norm_(vision_student.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            progress.set_postfix({'Loss': f"{loss.item():.4f}"})
            
        avg_loss = total_loss / len(train_loader)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({'model_state_dict': vision_student.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, os.path.join(save_dir, "phase2_vision_student_best.pth"))
            
        metrics = evaluate_vision_audio_retrieval(vision_student, audio_teacher, val_loader, device)
        print(f"Phase 2 A2V R@1: {metrics['A2V']['R@1']:.3f} | V2A R@1: {metrics['V2A']['R@1']:.3f}")

if __name__ == '__main__':
    main()