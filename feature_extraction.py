import librosa
import numpy as np

from config import N_MFCC


def extract_mfcc(audio, sr):

    """
    Extract MFCC Features
    """

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=N_MFCC
    )

    return mfcc


def extract_mel_spectrogram(audio, sr):

    """
    Extract Mel Spectrogram
    """

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr
    )

    mel_db = librosa.power_to_db(
        mel,
        ref=np.max
    )

    return mel_db


def extract_chroma(audio, sr):

    """
    Extract Chroma Features
    """

    chroma = librosa.feature.chroma_stft(
        y=audio,
        sr=sr
    )

    return chroma