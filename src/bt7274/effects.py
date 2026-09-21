"""
Audio effects for BT's voice.
"""

import numpy as np

_rng = np.random.default_rng()


def radio_filter(
    audio: np.ndarray,
    sample_rate: int,
    low_hz: float = 300.0,
    high_hz: float = 3800.0,
    drive: float = 2.5,
    hiss: float = 0.004,
) -> np.ndarray:
    """
    Make BT sound like he's coming over comms: band-limit the voice, add a little
    saturation and a touch of static. Works on a whole sentence at a time.
    """
    if audio.size == 0:
        return audio

    # Band-pass in the frequency domain with smooth 2nd-order-style roll-offs.
    spectrum = np.fft.rfft(audio)
    freqs = np.fft.rfftfreq(audio.size, d=1.0 / sample_rate)
    highpass = 1.0 / np.sqrt(1.0 + (low_hz / np.maximum(freqs, 1e-3)) ** 4)
    lowpass = 1.0 / np.sqrt(1.0 + (freqs / high_hz) ** 4)
    filtered = np.fft.irfft(spectrum * highpass * lowpass, n=audio.size)

    # Soft clipping gives the slightly crunchy radio edge.
    peak = np.max(np.abs(filtered)) or 1.0
    shaped = np.tanh(drive * filtered / peak) / np.tanh(drive)
    shaped += hiss * _rng.standard_normal(shaped.size)

    return (0.9 * shaped / (np.max(np.abs(shaped)) or 1.0)).astype(np.float32)
