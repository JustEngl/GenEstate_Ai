# Genestate AI - House Image Generator

AI-powered house visualization tool using Stable Diffusion XL.

## Features
- Generate house images based on style, size, roof color
- Real-time price estimation
- Multiple architectural styles (Modern, Wooden, Mediterranean)

## Setup

### Prerequisites
- Python 3.10+
- CUDA-capable GPU (8GB+ VRAM recommended), supporting CUDA 11.8
- ~15GB disk space for model weights
IMPORTANT: The LoRA weights are not included in this repo. I will provide a link to them later on. The model quality might be not as expected without the weights
### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/genestate-ai.git
cd genestate-ai
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```
4. Install CUDA-specific PyTorch packages:
```bash   
pip install torch==2.3.1+cu118 torchvision==0.18.1+cu118 torchaudio --index-url https://download.pytorch.org/whl/cu118
```
5. Install xFormers (required for faster generation):
```
pip install "https://download.pytorch.org/whl/cu118/xformers-0.0.27%2Bcu118-cp311-cp311-win_amd64.whl"
```

6. (OPTIONAL) Download model weights:
The base model downloads automatically. For the LoRA weights, you need to:
- Download from ... (coming soon)
- Place them in `lora_weights/genestate_lora_final/`

### Running
```bash
python app.py
```

Open browser to `http://localhost:5000`

## Tech Stack
- **Backend**: Flask, PyTorch
- **Model**: Dreamshaper XL Turbo + Custom LoRA
- **Frontend**: Vanilla JavaScript, HTML5, CSS3


## License
MIT

## Credits

Created by Justus Engel / EngelData

