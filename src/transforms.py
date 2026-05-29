import torch
import torchaudio
import torchvision.transforms as T
from PIL import Image

class MSMDSpectrogramPipeline(torch.nn.Module):
    def __init__(self, filterbank):
        super().__init__()
        self.n_fft = 2048
        self.hop_length = 1102
        self.register_buffer('filterbank', filterbank)
        self.spectrogram = torchaudio.transforms.Spectrogram(
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            power=1.0
        )

    def forward(self, waveform):
        mag_spec = self.spectrogram(waveform)
        filtered_spec = torch.einsum('bft,fm->bmt', mag_spec, self.filterbank)
        log_spec = torch.log10(filtered_spec + 1.0)
        return log_spec

def letterbox_image(pil_img, target_w=416, target_h=128, pad_color=(255, 255, 255)):
    orig_w, orig_h = pil_img.size
    scale = min(target_w / orig_w, target_h / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    resized = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (target_w, target_h), color=pad_color)
    paste_x = (target_w - new_w) // 2
    paste_y = (target_h - new_h) // 2
    canvas.paste(resized, (paste_x, paste_y))
    return canvas

def get_vision_augmenter():
    return T.Compose([
        T.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.RandomAffine(degrees=1, translate=(0.02, 0.02))
    ])