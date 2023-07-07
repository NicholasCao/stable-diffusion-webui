from basicsr.utils.download_util import load_file_from_url
import os

model_urls = [
    'https://huggingface.co/XpucT/Deliberate/resolve/main/Deliberate_v2.safetensors',
    'https://huggingface.co/dreamlike-art/dreamlike-photoreal-2.0/resolve/main/dreamlike-photoreal-2.0.safetensors',
    'https://huggingface.co/SG161222/Realistic_Vision_V1.4/resolve/main/Realistic_Vision_V1.4-pruned-fp16.safetensors',
    'https://huggingface.co/stabilityai/stable-diffusion-2-1/resolve/main/v2-1_768-ema-pruned.safetensors'
]

for u in model_urls:
    print(os.path.basename(u))
    dl = load_file_from_url(u, 'models/Stable-diffusion', True, os.path.basename(u))