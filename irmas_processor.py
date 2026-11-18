import os
import numpy as np
import librosa

from preprocessing import preprocess_audio

from segmentation import split_audio

from feature_extraction import extract_mfcc

from config import SAMPLE_RATE
from config import INSTRUMENTS


# IRMAS Dataset Path
dataset_path = r"dataset\IRMAS"

# Storage
X = []
y = []


# Process Each Instrument Folder
for instrument in os.listdir(dataset_path):

    instrument_path = os.path.join(
        dataset_path,
        instrument
    )

    # Skip non-folders
    if not os.path.isdir(instrument_path):
        continue

    print(f"\nProcessing Instrument: {instrument}")

    # Skip unknown instruments
    if instrument not in INSTRUMENTS:
        print(f"Skipping {instrument}")
        continue

    # Instrument Index
    instrument_index = INSTRUMENTS.index(
        instrument
    )

    # Process Audio Files
    for file_name in os.listdir(instrument_path):

        if file_name.endswith(".wav"):

            file_path = os.path.join(
                instrument_path,
                file_name
            )

            try:

                # Load Audio
                audio, sr = librosa.load(
                    file_path,
                    sr=SAMPLE_RATE,
                    mono=True
                )

                # Preprocess
                audio = preprocess_audio(audio)

                # Split Audio
                segments, timestamps = split_audio(
                    audio
                )

                # Process Segments
                for segment in segments:

                    try:

                        # Extract MFCC
                        mfcc = extract_mfcc(
                            segment,
                            SAMPLE_RATE
                        )

                        # Fixed Shape
                        mfcc = mfcc[:, :173]

                        # Skip small segments
                        if mfcc.shape[1] < 173:
                            continue

                        X.append(mfcc)

                        # Create Multi-Label Vector
                        label = [0] * len(INSTRUMENTS)

                        label[instrument_index] = 1

                        y.append(label)

                    except:
                        continue

            except Exception as e:

                print(f"File Error: {e}")

                continue


# Convert Arrays
X = np.array(X)
y = np.array(y)

# Add CNN Channel Dimension
X = X[..., np.newaxis]

print("\nIRMAS Dataset Created Successfully!")

print("X Shape:", X.shape)

print("y Shape:", y.shape)

# Save IRMAS Features
np.save(
    "extracted_features/X_irmas.npy",
    X
)

np.save(
    "extracted_features/y_irmas.npy",
    y
)

print("\nIRMAS Features Saved Successfully!")