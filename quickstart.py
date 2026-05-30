import torch
from huggingface_hub import hf_hub_download
from src.models import SheetMusicSwin, SpectrogramSwin

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 1. Initialize the encoders
vision_encoder = SheetMusicSwin().to(device)
audio_encoder = SpectrogramSwin().to(device)

# 2. Download and load pretrained weights
vision_path = hf_hub_download(repo_id="hyg444/SheetAudio-GCL", filename="SheetAudio-GCL-Sheet.pth")
audio_path = hf_hub_download(repo_id="hyg444/SheetAudio-GCL", filename="SheetAudio-GCL-Audio.pth")

vision_encoder.load_state_dict(torch.load(vision_path, map_location=device))
audio_encoder.load_state_dict(torch.load(audio_path, map_location=device))

vision_encoder.eval()
audio_encoder.eval()