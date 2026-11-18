import numpy as np
import numpy as np

import matplotlib
matplotlib.use("Agg")      # Use non-GUI backend

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau

from tensorflow.keras.metrics import BinaryAccuracy
from tensorflow.keras.metrics import Precision
from tensorflow.keras.metrics import Recall


# ==========================
# Load Dataset
# ==========================

X = np.load(
    "extracted_features/X_combined.npy"
)

y = np.load(
    "extracted_features/y_combined.npy"
)

print("X Shape:", X.shape)
print("y Shape:", y.shape)


# ==========================
# Train Test Split
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    shuffle=True
)


# ==========================
# CNN MODEL
# ==========================

model = Sequential()

# Block 1
model.add(
    Conv2D(
        32,
        (3,3),
        padding='same',
        activation='relu',
        input_shape=(40,173,1)
    )
)

model.add(BatchNormalization())

model.add(
    MaxPooling2D((2,2))
)

# Block 2
model.add(
    Conv2D(
        64,
        (3,3),
        padding='same',
        activation='relu'
    )
)

model.add(BatchNormalization())

model.add(
    MaxPooling2D((2,2))
)

# Block 3
model.add(
    Conv2D(
        128,
        (3,3),
        padding='same',
        activation='relu'
    )
)

model.add(BatchNormalization())

model.add(
    MaxPooling2D((2,2))
)

# Flatten
model.add(
    Flatten()
)

# Dense Layers
model.add(
    Dense(
        256,
        activation='relu'
    )
)

model.add(
    Dropout(0.5)
)

model.add(
    Dense(
        128,
        activation='relu'
    )
)

model.add(
    Dropout(0.3)
)

# Output Layer
model.add(
    Dense(
        14,
        activation='sigmoid'
    )
)


# ==========================
# Compile Model
# ==========================

model.compile(

    optimizer='adam',

    loss='binary_crossentropy',

    metrics=[
        BinaryAccuracy(name="binary_accuracy"),
        Precision(name="precision"),
        Recall(name="recall")
    ]
)

# Summary
model.summary()


# ==========================
# Callbacks
# ==========================

early_stop = EarlyStopping(

    monitor='val_loss',

    patience=10,

    restore_best_weights=True

)

checkpoint = ModelCheckpoint(

    "models/best_model.h5",

    monitor='val_loss',

    save_best_only=True,

    verbose=1

)

reduce_lr = ReduceLROnPlateau(

    monitor='val_loss',

    factor=0.5,

    patience=5,

    verbose=1

)


# ==========================
# Train Model
# ==========================

history = model.fit(

    X_train,

    y_train,

    epochs=80,

    batch_size=32,

    validation_data=(X_test, y_test),

    callbacks=[

        early_stop,

        checkpoint,

        reduce_lr

    ],

    shuffle=True

)


# ==========================
# Save Model
# ==========================

model.save(

    "models/instrument_model.h5"

)

print("\nModel Trained Successfully!")

print("Model Saved Successfully!")
# ==========================
# Generate Accuracy Graph
# ==========================

plt.figure(figsize=(8,6))

plt.plot(
    history.history['binary_accuracy'],
    label='Training Accuracy',
    linewidth=2
)

plt.plot(
    history.history['val_binary_accuracy'],
    label='Validation Accuracy',
    linewidth=2
)

plt.title("Training vs Validation Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend()

plt.grid(True)

plt.savefig(
    "accuracy_graph.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ==========================
# Generate Loss Graph
# ==========================

plt.figure(figsize=(8,6))

plt.plot(
    history.history['loss'],
    label='Training Loss',
    linewidth=2
)

plt.plot(
    history.history['val_loss'],
    label='Validation Loss',
    linewidth=2
)

plt.title("Training vs Validation Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()

plt.grid(True)

plt.savefig(
    "loss_graph.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ==========================
# Generate Precision & Recall Graph
# ==========================

plt.figure(figsize=(8,6))

plt.plot(
    history.history['precision'],
    label='Training Precision',
    linewidth=2
)

plt.plot(
    history.history['val_precision'],
    label='Validation Precision',
    linewidth=2
)

plt.plot(
    history.history['recall'],
    label='Training Recall',
    linewidth=2
)

plt.plot(
    history.history['val_recall'],
    label='Validation Recall',
    linewidth=2
)

plt.title("Precision and Recall")

plt.xlabel("Epoch")

plt.ylabel("Score")

plt.legend()

plt.grid(True)

plt.savefig(
    "precision_recall_graph.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nGraphs Generated Successfully!")  
# ==========================
# Generate Confusion Matrix
# ==========================

print("\nGenerating Confusion Matrix...")

# Predict on test data
y_pred = model.predict(X_test)

# Convert probabilities to binary predictions
y_pred = (y_pred >= 0.5).astype(int)

# Generate confusion matrix for each instrument
cm = multilabel_confusion_matrix(y_test, y_pred)

# Instrument names (update if your label order is different)
instrument_names = [
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
    "Voice",
    "Instrument12",
    "Instrument13",
    "Instrument14"
]

# Save confusion matrix for each instrument
for i in range(len(cm)):

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm[i],
        display_labels=["Absent", "Present"]
    )

    disp.plot(cmap="Blues", values_format="d")

    plt.title(f"Confusion Matrix - {instrument_names[i]}")

    plt.savefig(
        f"confusion_matrix_{instrument_names[i]}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

print("Confusion Matrices Generated Successfully!")           