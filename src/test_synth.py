import torch
import torchaudio
from synth import additive_synth

sample_rate = 16000
hop_size = 64

# Number of frames = how many pitch/loudness snapshots per second
# 16000 samples / 64 samples per frame = 250 frames per second
n_frames = sample_rate // hop_size
n_harmonics = 8

# f0: constant 440Hz across all 250 frames
# shape: (1, batch, 250 frames)
f0 = torch.full((1, n_frames), 440.0)

# Amplitude for each of the 8 harmonics
# Fundamental (1st) is loudest, each overtone gets quieter
harmonic_amps = torch.tensor([1.0, 0.5, 0.25, 0.12, 0.06, 0.03, 0.01, 0.01])

# Expand to (1, 250, 8) — same amplitudes for every frame
amplitudes = harmonic_amps.unsqueeze(0).unsqueeze(0).expand(1, n_frames, n_harmonics)

# Normalize so all harmonics sum to 1.0 per frame — prevents clipping
amplitudes = amplitudes / amplitudes.sum(dim=-1, keepdim=True)

# Generate audio
audio = additive_synth(f0, amplitudes, sample_rate, hop_size)

print(f"Output shape: {audio.shape}")  # expect (1, 16000)
print(f"Min: {audio.min():.4f}, Max: {audio.max():.4f}")

torchaudio.save("test_output.wav", audio, sample_rate)
print("Saved to test_output.wav")
