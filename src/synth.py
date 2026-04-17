import torch
import math


def additive_synth(f0, amplitudes, sample_rate=16000, hop_size=64):
    """
    f0:           (batch, n_frames)                     fundamental frequency in Hz per frame
    amplitude:    (batch, n_frames, n_harmonics)        amplitude of each harmonic per frame
    returns:      (batch, n_samples)                    final audio signal
    """

    # batch_size -> parallel audio signals (Process multiple examples in one forward pass)
    # n_frames -> time steps (low resolution) (this requires upsampling into samples later on)
    # n_harmmonics -> number of sine waves per frame
    batch_size, n_frames, n_harmonics = amplitudes.shape

    # Total number of audio samples in the output
    n_samples = n_frames * hop_size

    # Harmonic multipliers: [1, 2, 3, ..., n_harmonics]
    # If f0=440Hz, harmonics are 440, 880, 1320, 1760...
    harmonic_numbers = torch.arange(1, n_harmonics + 1, dtype=torch.float32)

    # Expand f0 from (batch, n_frames) to (batch, n_frames, n_harmonics)
    # so each frame has n_harmonics frequency values (f0*1, f0*2, f0*3...)
    # (batch, n_frames, 1) * (harmonic_numbers,) -> (batch, n_frames, harmonic_numbers)
    f0_expanded = f0.unsqueeze(-1) * harmonic_numbers

    # We have 250 frames but 16000 samples
    # Each frame needs to expand to cover hop_size (64) samples
    # repeat_interleave repeats each frame value 64 times along dim=1
    # (batch, n_frames, n_harmonics) -> (batch, n_samples, n_harmonics)
    f0_samples = f0_expanded.repeat_interleave(hop_size, dim=1)
    amplitudes_samples = amplitudes.repeat_interleave(hop_size, dim=1)
