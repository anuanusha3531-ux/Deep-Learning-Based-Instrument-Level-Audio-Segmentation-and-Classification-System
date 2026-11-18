import stempeg
import librosa
import numpy as np

from config import SAMPLE_RATE


def load_stem_audio(file_path):

    """
    Load MUSDB18 STEM file
    """

    # Read STEM file
    audio, sr = stempeg.read_stems(file_path)

    return audio, sr


def get_mixture_audio(audio):

    """
    Channel 0 = Full mixed song
    """

    mixture = audio[0]

    return mixture


def get_individual_sources(audio):

    """
    Extract instruments from channels
    """

    sources = {
        "Drums": audio[1],
        "Bass": audio[2],
        "Other": audio[3],
        "Vocals": audio[4]
    }

    return sources


def preprocess_audio(audio):

    """
    Convert stereo → mono
    Normalize audio
    """

    # Stereo to mono
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    # Normalize
    audio = audio / np.max(np.abs(audio))

    return audio