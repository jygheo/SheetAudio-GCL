# Graph Contrastive Learning for Sheet Music-Audio Alignment

This repository implements a cross-modal retrieval system mapping piano sheet music lines directly to corresponding audio spectrograms. Instead of using modality-specific pretraining, such as contrasting augmented views of data snippets ([Chaudhuri et al., 2023](https://arxiv.org/pdf/2309.12134)), we use a Graph Attention Network (GAT) trained on graph representations constructed from MSMD annotations ([Dorfer et al., 2018](https://transactions.ismir.net/articles/10.5334/tismir.12)) as a structural reference for alignment. We first pretrain the audio encoder contrastively against the GAT, then train the vision encoder to match the frozen GAT embeddings. This initializes both encoders against the same graph-based representation before direct cross-modal fine-tuning.

## Setup

### Clone the Repository

```bash
git clone https://github.com/jygheo/SheetAudio-GCL.git
cd SheetAudio-GCL
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Download the Dataset

Run the setup script to download the MSMD dataset and initialize the required directories.

```bash
bash setup.sh
```

### Model Usage

To use the models, see [quickstart.py](https://github.com/jygheo/SheetAudio-GCL/blob/main/quickstart.py).  
To reproduce results, see [Reproducing Results](#reproducing-results).

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

Training is performed in three steps to align graph, audio, and vision representations.

#### 1. Audio-Graph Pretraining

<img width="576" height="300" alt="Sheet_Audio_GCL_imgs_page-0001" src="https://github.com/user-attachments/assets/55040af5-07f0-47d8-b3cf-9d5f362737c1" />  

We use symmetric Momentum Contrast (MoCo) to align audio spectrogram embeddings with their corresponding graph embeddings produced by the Graph Attention Network (GAT).

#### 2. Vision-Graph Pretraining

<img width="380" height="300" alt="Sheet_Audio_GCL_imgs_page-0002" src="https://github.com/user-attachments/assets/ad807c54-e944-47db-8768-c0635405b3bf" />  

With the GAT frozen, we train the vision encoder to match the corresponding GAT embeddings from sheet music images.

#### 3. Vision-Audio Fine-Tuning

<img width="583" height="300" alt="Sheet_Audio_GCL_imgs_page-0003" src="https://github.com/user-attachments/assets/4408d955-60c2-4cdc-a126-8052d0f8ac5a" />  

The pretrained vision and audio encoders are jointly fine-tuned using symmetric MoCo to directly align sheet music images with audio spectrograms.

## Reproducing Results

Run training scripts in order:

### 1. Audio-Graph Pretraining
```bash
python scripts/01_train_phase1.py
```

### 2. Vision-Graph Pretraining
```bash
python scripts/02_train_phase2.py
```

### 3. Joint Vision-Audio Fine-Tuning
```bash
python scripts/03_train_phase3.py
```

### 4. GrandStaff Domain Adaptation
```bash
python scripts/04_finetune.py
```

### MSMD evaluation (using models from [Hugging Face](https://huggingface.co/hyg444/SheetAudio-GCL))
```bash
python scripts/05_evaluate_msmd.py
```

### GrandStaff evaluation
```bash
python scripts/06_eval_gs.py
```

## Results and Evaluation

Retrieval performance is evaluated using the following metrics:

- **Rank@K (R@1, R@25):** Proportion of queries for which the correct match appears within the top K retrieved results.
- **Mean Reciprocal Rank (MRR):** Mean reciprocal rank of the first correct match.
- **Median Rank (MR):** Median rank position of the correct match across all queries.

### 1. MSMD Dataset

Evaluation was conducted on the MSMD test split after direct vision-audio fine-tuning.

- **Unique sheet music lines:** 1,636  
- **Audio-vision query pairs:** 24,540

| Retrieval Direction | R@1 | R@25 | MRR | MR |
|:--------------------|----:|-----:|----:|---:|
| Audio-to-Vision     | 0.670 | 0.925 | 0.756 | 1.0 |
| Vision-to-Audio     | 0.733 | 0.895 | 0.779 | 1.0 |

### 2. GrandStaff Dataset

The MSMD-trained model was further fine-tuned and evaluated on the test split of the [GrandStaff multimodal](https://huggingface.co/datasets/PRAIG/grandstaff-grandstaff-multimodal) dataset. These results indicate that cross-modal alignment learned through the MSMD-specific pretraining can transfer to a different synthetic piano score-audio dataset.

- **Unique sheet music lines:** 7,793

| Retrieval Direction | R@1 | R@25 | MRR | MR |
|:--------------------|----:|-----:|----:|---:|
| Audio-to-Vision     | 0.880 | 0.990 | 0.924 | 1.0 |
| Vision-to-Audio     | 0.889 | 0.990 | 0.929 | 1.0 |

