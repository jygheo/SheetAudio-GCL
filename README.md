# Graph Contrastive Learning for Sheet Music-Audio Alignment

This repository implements a cross-modal retrieval system mapping piano sheet music lines directly to corresponding audio spectrograms. Instead of using modality-specific pretraining, such as contrasting augmented views of data snippets ([Chaudhuri et al., 2023](https://arxiv.org/pdf/2309.12134)), we use a Graph Attention Network (GAT) trained on graph representations constructed from MSMD annotations ([Dorfer et al., 2018](https://transactions.ismir.net/articles/10.5334/tismir.12)) as a structural reference for alignment. We first pretrain the audio encoder contrastively against the GAT, then train the vision encoder to match the frozen GAT embeddings. This initializes both encoders relative to the same structural reference before directly fine-tuning them against each other.

## Methodology
### Graph Construction

We construct graph representations of musical structure using annotations from the MSMD dataset.

#### Node Features

Each node corresponds to a musical note and contains:

- **Spatial features:** Bounding box coordinates extracted from the sheet music image (`top`, `left`, `width`, `height`)
- **Temporal features:** MIDI onset time and duration
- **Pitch:** MIDI pitch code

#### Edge Construction

Edges represent temporal relationships between notes:

- **Chord edges:** Bidirectional edges between all notes sharing the same onset time
- **Sequence edges:** Directed edges connecting all notes at a given onset to all notes at the next chronological onset

### Training 

Training is performed in three stages to  align graph, audio, and vision representations.

#### Stage 1: Audio-Graph Pretraining

<img width="576" height="300" alt="Sheet_Audio_GCL_imgs_page-0001" src="https://github.com/user-attachments/assets/55040af5-07f0-47d8-b3cf-9d5f362737c1" />  

We use symmetric Momentum Contrast (MoCo) to align audio spectrogram embeddings with their corresponding graph embeddings produced by the Graph Attention Network (GAT).

#### Stage 2: Vision-Graph Pretraining

<img width="380" height="300" alt="Sheet_Audio_GCL_imgs_page-0002" src="https://github.com/user-attachments/assets/ad807c54-e944-47db-8768-c0635405b3bf" />  

With the GAT frozen, we train the vision encoder to match the corresponding GAT embeddings from sheet music images.

#### Stage 3: Vision-Audio Fine-Tuning

<img width="583" height="300" alt="Sheet_Audio_GCL_imgs_page-0003" src="https://github.com/user-attachments/assets/4408d955-60c2-4cdc-a126-8052d0f8ac5a" />  

The pretrained vision and audio encoders are jointly fine-tuned using symmetric MoCo to directly align sheet music images with audio spectrograms.

## Results 

Retrieval performance is evaluated on the MSMD test split using Rank@1, Rank@25, Mean Reciprocal Rank (MRR), and Median Rank (MR) after 1 epoch of direct Vision-Audio fine-tuning.
MSMD one epoch
* **Audio-to-Vision:** R@1: 0.670 | R@25: 0.925 | MRR: 0.756 | MR: 1.0
* **Vision-to-Audio:** R@1: 0.733 | R@25: 0.895 | MRR: 0.779 | MR: 1.0
