'''

'''

import numpy as np
import logging
import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd import gradcheck
import speechbrain as sb
from diffwave.inferencefromt import predict as diffwave_predict
import torchaudio.transforms as TT


logger = logging.getLogger(__name__)


class DDPM(nn.Module):
    def __init__(self, sigma, model_dir, sr, filter=None):
        self.sigma = sigma
        self.model_dir = model_dir
        self.sr = sr
        if filter is not None:
            self.filter = filter(sigma=sigma)
        else:
            self.filter = None


    """
    Function to convert waveform to spectrogram. 
    To be used for conditional inference only.
    """
    def transform(self, sigs):
        audio, sr = sigs, self.sr
        audio = torch.clamp(audio[0], -1.0, 1.0)

        mel_args = {
            'sample_rate': sr,
            'win_length': 256 * 4,
            'hop_length': 256,
            'n_fft': 1024,
            'f_min': 20.0,
            'f_max': sr / 2.0,
            'n_mels': 80,
            'power': 1.0,
            'normalized': True,
        }
        mel_spec_transform = TT.MelSpectrogram(**mel_args)

        with torch.no_grad():
            spectrogram = mel_spec_transform(audio.cpu())
            spectrogram = spectrogram.to('cuda:0')
            spectrogram = 20 * torch.log10(torch.clamp(spectrogram, min=1e-5)) - 20
            spectrogram = torch.clamp((spectrogram + 100) / 100, 0.0, 1.0)
            
            return spectrogram


    def forward(self, sigs, sig_lens):
        """
        Pass the input through the DDPM model
        """

        # Pass to DDPM
        audio = sigs
        audio, sample_rate = diffwave_predict(audio, self.model_dir, fast_sampling=True, sigma=self.sigma)

        if self.filter:
            audio = self.filter(audio)
        
        return audio

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)