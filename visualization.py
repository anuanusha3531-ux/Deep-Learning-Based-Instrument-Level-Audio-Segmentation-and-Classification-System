import librosa.display
import matplotlib.pyplot as plt


def plot_mfcc(mfcc, save_path):

    plt.figure(figsize=(12, 4))

    librosa.display.specshow(
        mfcc,
        x_axis='time'
    )

    plt.colorbar()

    plt.title("MFCC")

    plt.tight_layout()

    plt.savefig(save_path)

    plt.close()


def plot_spectrogram(spectrogram, sr, save_path):

    plt.figure(figsize=(12, 4))

    librosa.display.specshow(
        spectrogram,
        sr=sr,
        x_axis='time',
        y_axis='mel'
    )

    plt.colorbar(format='%+2.0f dB')

    plt.title("Mel Spectrogram")

    plt.tight_layout()

    plt.savefig(save_path)

    plt.close()