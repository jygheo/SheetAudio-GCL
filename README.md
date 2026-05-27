# Graph Contrastive Learning for Sheet Music-Audio Alignment

This repository implements a cross-modal retrieval system mapping sheet music images directly to audio spectrograms. Instead of using standard modality-specific contrastive pretraining, such as contrasting augmented views of data snippets ([Chaudhuri et al., 2023](https://arxiv.org/pdf/2309.12134)), we utilize a Graph Attention Network (GAT) as a pretraining proxy. We construct a graph representation using annotations from the MSMD dataset ([Dorfer et al., 2018](https://transactions.ismir.net/articles/10.5334/tismir.12)). By aligning both the audio and vision encoders to the graph embeddings first, we establish a shared semantic space before directly fine-tuning the modalities against each other. 


