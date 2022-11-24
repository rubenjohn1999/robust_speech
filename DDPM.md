# Instructions to experiment with Diffwave

Install the required packages

```
git clone https://github.com/rubenjohn1999/robust_speech.git
git checkout ddpm-diffwave
cd robust_speech
pip install .
cd robust_speech/robust_speech/adversarial/defenses/diffwave
pip install .
```

- Be sure to download the Librispeech corpus into path/to/results/folder/data
- The file robust_speech/robust_speech/adversarial/defenses/ddpm.py acts as the entrypoint to the DDPM
- The model file robust_speech/robust_speech/models/wav2vec2_fine_tune.py has been modified to add a check for DDPM and subsequently the robust_speech/recipes/model_configs/wav2vec2-large-960h.yaml has been modified to include the reference to the DDPM
- The Diffwave model has been added at robust_speech/robust_speech/adversarial/defenses/diffwave. The original Diffwave implementation repository is -> https://github.com/lmnt-com/diffwave
- The pretrained checkpoint to Diffwave is also present at robust_speech/recipes/ddpm_model
- Follow the usual robust_speech instructions to train wav2vec2-large-960h and run 
```
python evaluate.py attack_configs/LibriSpeech/none/w2v2_large_960h.yaml --root=/path/to/results/folder
```