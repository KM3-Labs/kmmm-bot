import torch
from transformers import AutoConfig

# from app.core.config import settings  TODO replace with system mentioned in readme

from pathlib import Path

def get_dtype(device: torch.device):
    model_dtype = torch.float32
    if device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda')
            model_dtype = torch.float16
        else:
            device = torch.device('cpu')
            model_dtype = torch.float32
    else:
        if device == 'cuda':
            model_dtype = torch.float16
    return model_dtype

def is_decoder(config: AutoConfig):
    decoder_types = ['gpt2', 'gptj', 'gpt_neo', 'gpt_neox', 'xglm']
    encoder_types = ['distilbert', 'bert', 'xlm', 'xlm-roberta', 'roberta']

    if config.model_type in decoder_types:
        return True
    elif config.model_type in encoder_types:
        return False
    else:
        raise ValueError(f"Unknown model type: {config.model_type}")

def tensorized_path(model_name: str, path: str = None):  # Add check for S3 storage
    f = Path(path) / Path(model_name.split('/')[-1])
    return f, f.with_suffix('.model').exists()
