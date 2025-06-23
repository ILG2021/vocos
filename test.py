import torchaudio
num_samples = 16384
sampling_rate = 24000
train = True

y, sr = torchaudio.load("dataset_raw/segment_45.wav")

