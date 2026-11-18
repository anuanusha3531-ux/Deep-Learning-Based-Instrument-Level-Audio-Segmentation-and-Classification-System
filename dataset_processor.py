import os
import numpy as np

from preprocessing import load_stem_audio
from preprocessing import get_mixture_audio
from preprocessing import get_individual_sources
from preprocessing import preprocess_audio

from segmentation import split_audio

from feature_extraction import extract_mfcc

from config import SAMPLE_RATE


# MUSDB18 Dataset Path
dataset_path = r"dataset\\MUSDB18\\musdb18\\train"

# Feature and Label Storage
X = []
y = []

# Instrument Labels
instrument_labels = [
    "Drums",
    "Bass",
    "Other",
    "Vocals"
]


# Process all STEM files
for file_name in os.listdir(dataset_path):

    # Check STEM format
    if file_name.endswith(".stem.mp4"):

        file_path = os.path.join(
            dataset_path,
            file_name
        )

        print(f"Processing: {file_name}")

        try:

            # Load STEM file
            audio, sr = load_stem_audio(file_path)

            # Extract mixed audio
            mixture = get_mixture_audio(audio)

            # Preprocess mixture
            mixture = preprocess_audio(mixture)

            # Split mixture into segments
            segments, timestamps = split_audio(mixture)

            # Extract individual sources
            sources = get_individual_sources(audio)

            # Preprocess all sources first
            processed_sources = {}

            for instrument in instrument_labels:

                source_audio = preprocess_audio(
                    sources[instrument]
                )

                source_segments, _ = split_audio(
                    source_audio
                )

                processed_sources[instrument] = source_segments

            # Process each segment
            for idx, segment in enumerate(segments):

                try:

                    # Extract MFCC
                    mfcc = extract_mfcc(
                        segment,
                        SAMPLE_RATE
                    )

                    # Fixed width
                    mfcc = mfcc[:, :173]

                    # Skip short segments
                    if mfcc.shape[1] < 173:
                        continue

                    # Add channel dimension later
                    X.append(mfcc)

                    # Create multi-label vector
                    label = []

                    for instrument in instrument_labels:

                        source_segments = processed_sources[instrument]

                        if idx < len(source_segments):

                            source_segment = source_segments[idx]

                            # Energy Detection
                            energy = np.mean(
                                np.abs(source_segment)
                            )

                            # Threshold
                            if energy > 0.01:
                                label.append(1)
                            else:
                                label.append(0)

                        else:
                            label.append(0)

                    y.append(label)

                except Exception as e:

                    print(
                        f"Segment Error: {e}"
                    )

                    continue

        except Exception as e:

            print(
                f"File Error: {e}"
            )

            continue


# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

# Add CNN channel dimension
X = X[..., np.newaxis]

print("\nDataset Created Successfully!")

print("X Shape:", X.shape)

print("y Shape:", y.shape)

# Save Features
np.save(
    "extracted_features/X.npy",
    X
)

np.save(
    "extracted_features/y.npy",
    y
)

print("\nFeatures Saved Successfully!")