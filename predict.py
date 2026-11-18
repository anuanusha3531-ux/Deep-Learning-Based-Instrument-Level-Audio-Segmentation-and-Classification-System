import numpy as np
import librosa

from tensorflow.keras.models import load_model

from preprocessing import preprocess_audio

from segmentation import split_audio

from feature_extraction import extract_mfcc
from feature_extraction import extract_mel_spectrogram

from visualization import plot_mfcc
from visualization import plot_spectrogram

from config import SAMPLE_RATE
from config import THRESHOLD


# Instrument Labels
instrument_names = [

    "Drums",
    "Bass",
    "Vocals",

    "Cello",
    "Clarinet",
    "Flute",
    "Acoustic Guitar",
    "Electric Guitar",
    "Organ",
    "Piano",
    "Saxophone",
    "Trumpet",
    "Violin",
    "Voice"
]


# Load Trained Model
model = load_model(
    "models/instrument_model.h5"
)


def predict_instruments(file_path):

    print("\nLoading Audio...")

    # Load audio file
    audio, sr = librosa.load(
        file_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    # Preprocess audio
    audio = preprocess_audio(audio)

    # Split audio into segments
    segments, timestamps = split_audio(audio)

    print(f"\nTotal Segments: {len(segments)}")

    print("\nPredicting Instruments...\n")

    results = []

    # Process each segment
    for idx, segment in enumerate(segments):

        try:

            # Extract MFCC
            mfcc = extract_mfcc(
                segment,
                SAMPLE_RATE
            )

            # Fixed shape
            mfcc = mfcc[:, :173]

            # Skip invalid segments
            if mfcc.shape[1] < 173:
                continue

            # Reshape for CNN
            mfcc = mfcc[np.newaxis, ..., np.newaxis]

            # Predict
            prediction = model.predict(
                mfcc,
                verbose=0
            )[0]

            detected = []

            # Instrument Detection
            for i, prob in enumerate(prediction):

                if prob > THRESHOLD:

                    detected.append(
                        instrument_names[i]
                    )

            # Prevent empty prediction
            if len(detected) == 0:

                max_index = np.argmax(prediction)

                detected.append(
                    instrument_names[max_index]
                )

            # Get timestamp
            start, end = timestamps[idx]

            result = {

                "start": round(start, 2),

                "end": round(end, 2),

                "instruments": detected
            }

            results.append(result)

            print(
                f"{start:.2f}s - "
                f"{end:.2f}s "
                f"→ {detected}"
            )

        except Exception as e:

            print(
                f"Prediction Error: {e}"
            )

            continue

    # Generate Visualizations
    print("\nGenerating Graphs...")

    # Full MFCC
    full_mfcc = extract_mfcc(
        audio,
        sr
    )

    # Mel Spectrogram
    mel = extract_mel_spectrogram(
        audio,
        sr
    )

    # Save MFCC Graph
    plot_mfcc(
        full_mfcc,
        "static/mfcc/mfcc_prediction.png"
    )

    # Save Spectrogram
    plot_spectrogram(
        mel,
        sr,
        "static/spectrograms/spectrogram_prediction.png"
    )

    print("\nGraphs Generated Successfully!")

    return results


# TESTING
# TESTING
if __name__ == "__main__":

    file_path = r"dataset/IRMAS/pia/[pia][cla]1283__1.wav"

    results = predict_instruments(
        file_path
    )

    print("\nPrediction Completed!")