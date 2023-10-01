dependencies = ['torch', 'torchaudio', 'numpy']

import torch
import json
from pathlib import Path


from wavlm.WavLM import WavLM, WavLMConfig
from hifigan.models import Generator as HiFiGAN
from hifigan.utils import AttrDict
from matcher import KNeighborsVC


ckpt_base_url = "https://github.com/bshall/knn-vc/releases/download/v0.1"


def knn_vc(pretrained: bool = True, progress: bool = True, prematched: bool = True, device: str = 'cuda') -> KNeighborsVC:
    """Load kNN-VC inference model. Optionally use vocoder trained on `prematched` data."""

    hifigan, hifigan_cfg = hifigan_wavlm(pretrained, progress, prematched, device)
    wavlm = wavlm_large(pretrained, progress, device)
    knnvc = KNeighborsVC(wavlm, hifigan, hifigan_cfg, device)

    print(f"{"pretrained" if pretrained else "Initialized"} kNN-VC {"prematched" if prematched else "pure"} WavLM-L6 model is loaded.")

    return knnvc


def hifigan_wavlm(pretrained: bool = True, progress: bool = True, prematched: bool = True, device: str = 'cuda') -> HiFiGAN:
    """Load pretrained WavLM-L6 HiFi-GAN vocoder. Optionally use weights trained on `prematched` data."""

    # Load configs
    cp = Path(__file__).parent.absolute()
    with open(cp/'hifigan'/'config_v1_wavlm.json') as f:
        data = f.read()
    json_config = json.loads(data)
    h = AttrDict(json_config)
    device = torch.device(device)

    # Init
    model = HiFiGAN(h).to(device)
    
    # Load state
    if pretrained:
        ckpt_name = "prematch_g_02500000.pt" if prematched else "g_02500000.pt"
        url = f"{ckpt_base_url}/{ckpt_name}"
        state_dict_g = torch.hub.load_state_dict_from_url(url, map_location=device, progress=progress)
        model.load_state_dict(state_dict_g['generator'])

    # Switch modes
    model.eval()
    model.remove_weight_norm()

    return model, h


def wavlm_large(pretrained: bool = True, progress: bool = True, device: str = 'cuda') -> WavLM:
    """Load the WavLM large checkpoint from the original paper. See https://github.com/microsoft/unilm/tree/master/wavlm for details. """

    if torch.cuda.is_available() == False:
        if str(device) != 'cpu':
            print(f"Overriding device {device} to cpu since no GPU is available.")
            device = 'cpu'

    url_ckpt = f"{ckpt_base_url}/WavLM-Large.pt"
    checkpoint = torch.hub.load_state_dict_from_url(url_ckpt, map_location=device, progress=progress)
    cfg = WavLMConfig(checkpoint['cfg'])
    device = torch.device(device)

    # Init
    model = WavLM(cfg)

    # Load state
    if pretrained:
        model.load_state_dict(checkpoint['model'])

    # Switch modes
    model = model.to(device)
    model.eval()

    return model
