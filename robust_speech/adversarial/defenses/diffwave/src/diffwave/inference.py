# Copyright 2020 LMNT, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np
import os
import torch
import torchaudio

from argparse import ArgumentParser

from diffwave.params import AttrDict, params as base_params
from diffwave.model import DiffWave
import time


models = {}

def calculate_timestep(alpha, sigma):
  expected_alpha = 1/(1 + sigma**2)
  best_value = 0
  best_diff = float("inf")
  i = 0
  for ele in alpha:
    if ele-expected_alpha > 0 and ele-expected_alpha < best_diff:
        best_diff = ele-expected_alpha
        best_value = i
    i += 1
  return best_value

def predict(spectrogram=None, model_dir=None, params=base_params, device=torch.device('cuda'), fast_sampling=False, sigma=0.02):
  
  # Lazy load model.
  if not model_dir in models:
    if os.path.exists(f'{model_dir}/weights.pt'):
      checkpoint = torch.load(f'{model_dir}/weights.pt')
    else:
      checkpoint = torch.load(model_dir)
    model = DiffWave(AttrDict(base_params)).to(device)
    model.load_state_dict(checkpoint['model'], strict=False)
    model.eval()
    models[model_dir] = model

  model = models[model_dir]
  model.params.override(params)
  with torch.no_grad():
    # Change in notation from the DiffWave paper for fast sampling.
    # DiffWave paper -> Implementation below
    # --------------------------------------
    # alpha -> talpha
    # beta -> training_noise_schedule
    # gamma -> alpha
    # eta -> beta


    if not model.params.unconditional:
      if len(spectrogram.shape) == 2:# Expand rank 2 tensors by adding a batch dimension.
        spectrogram = spectrogram.unsqueeze(0)
      spectrogram = spectrogram.to(device)
      audio = torch.randn(spectrogram.shape[0], model.params.hop_samples * spectrogram.shape[-1], device=device)
      
    else:
      # audio = torch.randn(1, params.audio_len, device=device)

      # Instead of starting with noise, start with the waveform
      audio = spectrogram.to(device)
      spectrogram = None

    # torchaudio.save("waveform_before_forward_ddpm.wav", audio.cpu(), 22050)
    # If we want to add noise through the forward pass of the ddpm,
    # Forward pass the waveform through the ddpm
    # audio = forward_ddpm(audio, model)
    # torchaudio.save("waveform_after_forward_ddpm.wav", audio.cpu(), 22050)

    training_noise_schedule = np.array(model.params.noise_schedule)
    # fast_sampling is passed as True so that the inference_noise_schedule is used
    inference_noise_schedule = np.array(model.params.inference_noise_schedule) if fast_sampling else training_noise_schedule


    # Perform the reverse process to convert noise to waveform
    talpha = 1 - training_noise_schedule
    talpha_cum = np.cumprod(talpha)

    beta = inference_noise_schedule
    alpha = 1 - beta
    alpha_cum = np.cumprod(alpha)

    T = []
    for s in range(len(inference_noise_schedule)):
      for t in range(len(training_noise_schedule) - 1):
        if talpha_cum[t+1] <= alpha_cum[s] <= talpha_cum[t]:
          twiddle = (talpha_cum[t]**0.5 - alpha_cum[s]**0.5) / (talpha_cum[t]**0.5 - talpha_cum[t+1]**0.5)
          T.append(t + twiddle)
          break

    T = np.array(T, dtype=np.float32)
    
    t_star = calculate_timestep(alpha, sigma)
    # print("t* = ", t_star)
    # t_star = 1


    delta = np.random.normal(0, scale=sigma, size=audio.shape).astype(np.float32)
    delta = torch.tensor(delta).to(audio.device)

    audio = alpha[t_star]**0.5 * (audio + delta)
    audio = torch.clamp(audio, -1.0, 1.0)
    # torchaudio.save("waveform_after_rescaling.wav", audio.cpu(), 16000)

    noise_scale = torch.from_numpy(alpha_cum**0.5).float().unsqueeze(1).to(device)
    for n in range(t_star, -1, -1):
      c1 = 1 / alpha[n]**0.5
      c2 = beta[n] / (1 - alpha_cum[n])**0.5
      
      audio = c1 * (audio - c2 * model(audio, torch.tensor([T[n]], device=audio.device), spectrogram).squeeze(1))
      if n > 0:
        noise = torch.randn_like(audio)
        sigma = ((1.0 - alpha_cum[n-1]) / (1.0 - alpha_cum[n]) * beta[n])**0.5
        audio += sigma * noise
      audio = torch.clamp(audio, -1.0, 1.0)
      # filename = "waveform_in_ddpm_iter_" + str(n) + ".wav"
      # torchaudio.save(filename, audio.cpu(), 16000)
  
  #print("DONE")
  # time.sleep(10)

  # torchaudio.save("waveform_after_bacward_ddpm.wav", audio.cpu(), 16000)
  return audio, model.params.sample_rate


'''
This function is the forward process of the dppm, to convert waveform to noise. 
'''
def forward_ddpm(audio, model):

  model.eval()
  beta = np.array(model.params.noise_schedule)
  noise_level = np.cumprod(1 - beta)
  noise_level = torch.tensor(noise_level.astype(np.float32))
  device = audio.device
  noise_level = noise_level.to(device)
  spectrogram = None

  for t in range(len(model.params.noise_schedule)):
    noise_scale = noise_level[t].unsqueeze(0)
    noise_scale_sqrt = noise_scale**0.5
    noise = torch.randn_like(audio)
    noisy_audio = noise_scale_sqrt * audio + (1.0 - noise_scale)**0.5 * noise
    # Passing the diffusion time step as len(noise_level) - 1 - t to add minimal amount of noise
    audio = model(noisy_audio, torch.tensor([len(noise_level) - 1 - t], device=audio.device), spectrogram).squeeze(1)

  return audio

def main(args):
  if args.spectrogram_path:
    spectrogram = torch.from_numpy(np.load(args.spectrogram_path))
  else:
    spectrogram = None
  audio, sr = predict(spectrogram, model_dir=args.model_dir, fast_sampling=args.fast, params=base_params)
  torchaudio.save(args.output, audio.cpu(), sample_rate=sr)


if __name__ == '__main__':
  parser = ArgumentParser(description='runs inference on a spectrogram file generated by diffwave.preprocess')
  parser.add_argument('model_dir',
      help='directory containing a trained model (or full path to weights.pt file)')
  parser.add_argument('--spectrogram_path', '-s',
      help='path to a spectrogram file generated by diffwave.preprocess')
  parser.add_argument('--output', '-o', default='output.wav',
      help='output file name')
  parser.add_argument('--fast', '-f', action='store_true',
      help='fast sampling procedure')
  main(parser.parse_args())
