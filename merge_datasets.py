import numpy as np


# Load MUSDB18 Dataset
X_musdb = np.load(
    "extracted_features/X.npy"
)

y_musdb = np.load(
    "extracted_features/y.npy"
)


# Load IRMAS Dataset
X_irmas = np.load(
    "extracted_features/X_irmas.npy"
)

y_irmas = np.load(
    "extracted_features/y_irmas.npy"
)


print("MUSDB X:", X_musdb.shape)
print("MUSDB y:", y_musdb.shape)

print("IRMAS X:", X_irmas.shape)
print("IRMAS y:", y_irmas.shape)


# IMPORTANT
# MUSDB has fewer labels than IRMAS
# We expand MUSDB labels to match IRMAS label size

target_label_size = y_irmas.shape[1]

expanded_y_musdb = []

for label in y_musdb:

    expanded = [0] * target_label_size

    # Drums
    expanded[0] = label[0]

    # Bass
    expanded[1] = label[1]

    # Vocals
    expanded[2] = label[3]

    expanded_y_musdb.append(expanded)

expanded_y_musdb = np.array(
    expanded_y_musdb
)


print(
    "\nExpanded MUSDB Labels:",
    expanded_y_musdb.shape
)


# Merge Datasets
X_combined = np.concatenate(
    [X_musdb, X_irmas],
    axis=0
)

y_combined = np.concatenate(
    [expanded_y_musdb, y_irmas],
    axis=0
)


print("\nCombined Dataset:")

print("X Shape:", X_combined.shape)

print("y Shape:", y_combined.shape)


# Save Final Dataset
np.save(
    "extracted_features/X_combined.npy",
    X_combined
)

np.save(
    "extracted_features/y_combined.npy",
    y_combined
)

print(
    "\nCombined Dataset Saved Successfully!"
)