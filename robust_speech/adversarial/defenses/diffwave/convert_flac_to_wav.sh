
# For each file in .flac files directory
for f in $(find /home/ubuntu/robust_speech/recipes/results/data/LibriSpeech/train-clean-100 -type f -name "*.flac");
do 
    # Convert to .wav keeping the same file name
    ffmpeg -i $f ./wav/$(basename -- "$f" .flac).wav
done
