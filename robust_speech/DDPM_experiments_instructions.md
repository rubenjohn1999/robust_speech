Here is a set of instructions on how to use DDPM while running experiments for CW attack.

1. DDPM 

If we only want the DDPM to be present, your wav2vec2-large-960h.yaml (```/home/ubuntu/robust_speech/recipes/model_configs/wav2vec2-large-960h.yaml```) file should look like the one below. 

 The inference.py (```/home/ubuntu/robust_speech/robust_speech/adversarial/defenses/diffwave/src/diffwave/inference.py```) file has the predict function, the sigma parameter value in the function also needs to be changed to the value that we write in the yaml file.

Example: Say we want to run CW for sigma=0.02 for the configuration of only DDPM


The wav2vec2-large-960h.yaml block will look like this
```
sigma: 0.02
# smoothing: !new:robust_speech.adversarial.defenses.smoothing.SpeechNoiseAugmentation
#    sigma: !ref <sigma>
   # filter config
   # filter: !name:robust_speech.adversarial.defenses.filter.ASNRWiener
   #    sr: 16000
   #    hop: 160
   #    nfft: 1024
   # enhancer: null

ddpm: !new:robust_speech.adversarial.defenses.ddpm.DDPM
    sigma: !ref <sigma>
    # change this path based on directory structure
    model_dir: /home/ubuntu/robust_speech/recipes/ddpm_model/unconditional-weights-691998.pt
    sr: 16000
   #  filter: !name:robust_speech.adversarial.defenses.filter.ASNRWiener
   #    sr: 16000
   #    hop: 160
   #    nfft: 1024
   ```

   The predict function in inference.py will look like this

   ```
   def predict(spectrogram=None, model_dir=None, params=base_params, device=torch.device('cuda'), fast_sampling=False, sigma=0.02):
   ```


2. DDPM + wiener

If we want both DDPM and wiener to be there, then we need to uncomment the wiener filter block that is present near the DDPM in the yaml file and inference.py predict function sigma value passed should be the same as that of what we have in the yaml file.

```
sigma: 0.02
# smoothing: !new:robust_speech.adversarial.defenses.smoothing.SpeechNoiseAugmentation
#    sigma: !ref <sigma>
   # filter config
   # filter: !name:robust_speech.adversarial.defenses.filter.ASNRWiener
   #    sr: 16000
   #    hop: 160
   #    nfft: 1024
   # enhancer: null

ddpm: !new:robust_speech.adversarial.defenses.ddpm.DDPM
    sigma: !ref <sigma>
    # change this path based on directory structure
    model_dir: /home/ubuntu/robust_speech/recipes/ddpm_model/unconditional-weights-691998.pt
    sr: 16000
    filter: !name:robust_speech.adversarial.defenses.filter.ASNRWiener
      sr: 16000
      hop: 160
      nfft: 1024
```
```
   def predict(spectrogram=None, model_dir=None, params=base_params, device=torch.device('cuda'), fast_sampling=False, sigma=0.02):
```


Once you have made the changes, 

```
cd /home/ubuntu/robust_speech/robust_speech/adversarial/defenses/diffwave
pip install .
```
Ensure you do this to make sure that the changes made to inference.py (i.e. the sigma value) takes effect. 
Then go ahead an run the evaluate.py file as you would normally. 

NOTE : In both these configurations, the smoothing block is commented except for the line ```sigma: 0.02```, double check this before you go ahead and run. This is because we do smoothing in the DDPM itself.

Also ensure the path to the checkpoint file is correct in the yaml file!