
from pathlib import Path

import torch
from torch import nn, Tensor
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from torchaudio.sox_effects import apply_effects_tensor

from hifigan.models import Generator as HiFiGAN
from hifigan.utils import AttrDict
from wavlm.WavLM import WavLM
from utils import generate_matrix_from_index


# Config
SPEAKER_INFORMATION_LAYER: int = 6 # Layer number of transformer, from which WavLM feature is extracted


SPEAKER_INFORMATION_WEIGHTS = generate_matrix_from_index(SPEAKER_INFORMATION_LAYER)


def fast_cosine_dist(source_feats: Tensor, matching_pool: Tensor, device: str = 'cpu') -> Tensor:
    """Like torch.cdist, but fixed dim=-1 and for cosine distance.
    
    Args:
        source_feats
        matching_pool
        device
    Returns:
        dists
    """

    source_norms = torch.norm(source_feats, p=2, dim=-1).to(device)
    matching_norms = torch.norm(matching_pool, p=2, dim=-1)
    dotprod = -torch.cdist(source_feats[None].to(device), matching_pool[None], p=2)[0]**2 + source_norms[:, None]**2 + matching_norms[None]**2
    dotprod /= 2

    dists = 1 - ( dotprod / (source_norms[:, None] * matching_norms[None]) )

    return dists


class KNeighborsVC(nn.Module):
    """The kNN-VC inference model."""

    def __init__(self, wavlm: WavLM, hifigan: HiFiGAN, hifigan_cfg: AttrDict, device: str = 'cuda'):
        """Init.
        Args:
            wavlm        - trained WavLM model
            hifigan      - trained hifigan model
            hifigan_cfg  - hifigan config to use for vocoding.
        """

        super().__init__()

        # set which features to extract from wavlm
        self.weighting = torch.tensor(SPEAKER_INFORMATION_WEIGHTS, device=device)[:, None]
        self.hifigan = hifigan.eval()
        self.h = hifigan_cfg
        self.wavlm = wavlm.eval()
        self.device = torch.device(device)
        self.sr = self.h.sampling_rate
        self.hop_length = 320

    def get_matching_set(self, wavs: list[Path] | list[Tensor], weights=None, vad_trigger_level=7) -> Tensor:
        """Extract WavLM features as *matching set* from reference utterances.

        Args:
            wavs              :: path[] | (Channel, T)[] | (T,)[] - reference 16kHz waveforms
            weights                                               - custom WavLM feature weighting
            vad_trigger_level                                     -
        Returns:
                              :: (Frame, Feat)                    - WavLM features, all frames are concatenated
        """

        # Conversion :: path|Tensor[] -> (Frame, Feat)[]
        feats = []
        for wave_src in wavs:
            feats.append(self.get_features(wave_src, weights=self.weighting if weights is None else weights, vad_trigger_level=vad_trigger_level))

        # Reshape :: (Frame, Feat)[] -> (Frame, Feat) - Concatenate all frames
        matching_set = torch.concat(feats, dim=0).cpu()

        return matching_set

    @torch.inference_mode()
    def vocode(self, c: Tensor) -> Tensor:
        """Unit-to-Wave vocoding, (B, Frame, Feat) -> (B, 1, T) -> (B, T)."""
        return self.hifigan(c).squeeze(1)

    @torch.inference_mode()
    def get_features(self, wave_src: str | Path | Tensor, weights=None, vad_trigger_level: float = 0):
        """Extract WavLM feature series (Load-Resample-Trim-Extraction).

        Args:
            wave_src :: path | (Channel, T) | (T,) - Source of waveform, file path or raw Tensor
            weights
            vad_trigger_level                      - Silence trimming threshold
        Returns:
            :: (Frame, Feat)                       - WavLM feature series
        """
        # TODO: Check whether to properly accept multi-channel

        if weights == None:
            weights = self.weighting

        # Load :: -> (Channel, T)
        ## From the file
        if type(wave_src) in [str, Path]:
            x, sr = torchaudio.load(wave_src, normalize=True)
        ## From the tensor
        else:
            x: Tensor = wave_src
            sr = self.sr
            if x.dim() == 1:
                ## (T,) -> (Channel=1, T)
                x = x[None]
        
        # Resampling
        if not sr == self.sr :
            print(f"resample {sr} to {self.sr} in {wave_src}")
            x = torchaudio.functional.resample(x, orig_freq=sr, new_freq=self.sr)
            sr = self.sr
            
        # Trimming - Head/Tail silence trim
        if vad_trigger_level > 1e-3:
            transform = T.Vad(sample_rate=sr, trigger_level=vad_trigger_level)
            x_front_trim = transform(x)
            # original way, disabled because it lacks windows support
            #waveform_reversed, sr = apply_effects_tensor(x_front_trim, sr, [["reverse"]])
            waveform_reversed = torch.flip(x_front_trim, (-1,))
            waveform_reversed_front_trim = transform(waveform_reversed)
            waveform_end_trim = torch.flip(waveform_reversed_front_trim, (-1,))
            #waveform_end_trim, sr = apply_effects_tensor(waveform_reversed_front_trim, sr, [["reverse"]])
            x = waveform_end_trim

        # Feature extraction :: (Channel, T) -> (Frame, Feat)
        wav_input_16khz = x.to(self.device)
        if torch.allclose(weights, self.weighting):
            ## :: (Channel, T) -> (B=1, Frame, Feat) -> (Frame, Feat)
            features = self.wavlm.extract_features(wav_input_16khz, output_layer=SPEAKER_INFORMATION_LAYER, ret_layer_results=False)[0].squeeze(0)
        else:
            # use slower weighted
            rep, layer_results = self.wavlm.extract_features(wav_input_16khz, output_layer=self.wavlm.cfg.encoder_layers, ret_layer_results=True)[0]
            features = torch.cat([x.transpose(0, 1) for x, _ in layer_results], dim=0) # (n_layers, seq_len, dim)
            # save full sequence
            features = ( features*weights[:, None] ).sum(dim=0) # (seq_len, dim)
        
        return features

    @torch.inference_mode()
    def match(self,
        query_seq:       Tensor,
        matching_set:    Tensor,
        synth_set:       Tensor | None = None, 
        topk:            int           =    4,
        tgt_loudness_db: float  | None =  -16,
        device:          str    | None = None,
    ) -> Tensor:
        """Given `query_seq`, `matching_set`, and `synth_set` tensors of shape (N, dim), perform kNN regression matching with k=`topk`.
        
        Args:
            query_seq    :: (Frame=n1, dim) - input/source query features
            matching_set :: (Frame=n2, dim) - The matching set from target speaker's utterances
            synth_set    :: (Frame=n2, dim) - corresponding to the matching set. We use the matching set to assign each query
                                              vector to a vector in the matching set, and then use the corresponding vector from the synth set during HiFiGAN synthesis.
                                              By default, and for best performance, this should be identical to the matching set.
            topk                            - 'k' in the kNN, the number of nearest neighbors to average over.
            tgt_loudness_db                 - Target loudness, normalized to this value [dB]. None means no normalization.
            device                          - Device for tensors. if None, uses default device at initialization.
        Returns:
            -            :: (T,)            - converted waveform
        """

        # Preparation
        device = torch.device(device) if device is not None else self.device
        synth_set = matching_set if synth_set is None else synth_set
        ## Device
        synth_set, matching_set, query_seq = synth_set.to(device), matching_set.to(device), query_seq.to(device)

        # k-NN - distance/topK
        dists = fast_cosine_dist(query_seq, matching_set, device=device)
        best = dists.topk(k=topk, largest=False, dim=-1)
        out_feats = synth_set[best.indices].mean(dim=1)

        # Vocoding - unit-to-wave
        prediction = self.vocode(out_feats[None].to(device)).cpu().squeeze()

        # Volume normalization
        if tgt_loudness_db is not None:
            src_loudness = torchaudio.functional.loudness(prediction[None], self.h.sampling_rate)
            tgt_loudness = tgt_loudness_db
            pred_wav = torchaudio.functional.gain(prediction, tgt_loudness - src_loudness)
        else:
            pred_wav = prediction

        return pred_wav


