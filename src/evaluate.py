import torch
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

@torch.no_grad()
def evaluate_graph_audio_retrieval(moco_model, val_loader, device='cuda'):
    moco_model.eval()
    all_graph_embeddings, all_audio_embeddings, all_line_ids = [], [], []

    for batch_data in tqdm(val_loader, desc="Evaluation Forward Pass"):
        q_graph = moco_model.encoder_q_graph(
            x_cont=batch_data['graph_x_cont'].to(device),
            x_class=batch_data['graph_x_class'].to(device),
            x_pitch=batch_data['graph_x_pitch'].to(device),
            edge_index=batch_data['graph_edge_index'].to(device),
            batch=batch_data['graph_batch_index'].to(device)
        )
        q_audio = moco_model.encoder_q_audio(batch_data['spectrograms'].to(device))
        
        all_graph_embeddings.append(q_graph.cpu())
        all_audio_embeddings.append(q_audio.cpu())
        all_line_ids.extend(batch_data['line_id'])

    graph_embeds = torch.cat(all_graph_embeddings, dim=0)
    audio_embeds = torch.cat(all_audio_embeddings, dim=0)
    similarity_matrix = torch.matmul(audio_embeds, graph_embeds.T)
    
    labels_tensor = torch.tensor(LabelEncoder().fit_transform(all_line_ids))
    ground_truth_mask = (labels_tensor.unsqueeze(1) == labels_tensor.unsqueeze(0))

    def calc_ranks(sim_matrix):
        ranks = torch.gather(ground_truth_mask, 1, torch.argsort(sim_matrix, dim=1, descending=True)).float().argmax(dim=1) + 1.0
        return (ranks <= 1).float().mean().item(), (ranks <= 25).float().mean().item(), (1.0 / ranks).mean().item(), ranks.median().item()

    r1_a2s, r25_a2s, mrr_a2s, mr_a2s = calc_ranks(similarity_matrix)
    r1_s2a, r25_s2a, mrr_s2a, mr_s2a = calc_ranks(similarity_matrix.T)
    
    moco_model.train()
    return {'A2S': {'R@1': r1_a2s, 'R@25': r25_a2s, 'MRR': mrr_a2s, 'MR': mr_a2s},
            'S2A': {'R@1': r1_s2a, 'R@25': r25_s2a, 'MRR': mrr_s2a, 'MR': mr_s2a}}

@torch.no_grad()
def evaluate_vision_audio_retrieval(vision_encoder, audio_encoder, val_loader, device='cuda'):
    vision_encoder.eval()
    audio_encoder.eval()
    all_vision_embeddings, all_audio_embeddings, all_line_ids = [], [], []

    for batch_data in tqdm(val_loader, desc="Evaluation Forward Pass"):
        all_vision_embeddings.append(vision_encoder(batch_data['images'].to(device)).cpu())
        all_audio_embeddings.append(audio_encoder(batch_data['spectrograms'].to(device)).cpu())
        all_line_ids.extend(batch_data['line_id'])

    vision_embeds = torch.cat(all_vision_embeddings, dim=0)
    audio_embeds = torch.cat(all_audio_embeddings, dim=0)

    unique_line_ids, unique_vision_indices, seen = [], [], set()
    for idx, lid in enumerate(all_line_ids):
        if lid not in seen:
            seen.add(lid)
            unique_line_ids.append(lid)
            unique_vision_indices.append(idx)

    dedup_vision_embeds = vision_embeds[unique_vision_indices]
    similarity_matrix = torch.matmul(audio_embeds, dedup_vision_embeds.T)

    encoder = LabelEncoder()
    encoder.fit(unique_line_ids)
    ground_truth_mask = (torch.tensor(encoder.transform(all_line_ids)).unsqueeze(1) == torch.tensor(encoder.transform(unique_line_ids)).unsqueeze(0))

    def calc_ranks(sim_matrix, mask):
        ranks = torch.gather(mask, 1, torch.argsort(sim_matrix, dim=1, descending=True)).float().argmax(dim=1) + 1.0
        return (ranks <= 1).float().mean().item(), (ranks <= 25).float().mean().item(), (1.0 / ranks).mean().item(), ranks.median().item()

    r1_a2v, r25_a2v, mrr_a2v, mr_a2v = calc_ranks(similarity_matrix, ground_truth_mask)
    r1_v2a, r25_v2a, mrr_v2a, mr_v2a = calc_ranks(similarity_matrix.T, ground_truth_mask.T)

    vision_encoder.train()
    audio_encoder.train()
    return {'A2V': {'R@1': r1_a2v, 'R@25': r25_a2v, 'MRR': mrr_a2v, 'MR': mr_a2v},
            'V2A': {'R@1': r1_v2a, 'R@25': r25_v2a, 'MRR': mrr_v2a, 'MR': mr_v2a}}