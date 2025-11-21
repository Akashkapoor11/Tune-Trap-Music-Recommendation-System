# train_emotion.py
# Training script using transfer learning (MobileNetV2).
# Prepare your dataset in data/train/<label>/ and data/val/<label>/ folders.
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

IMG_SIZE = (224,224)
BATCH = 32
EPOCHS = 15
NUM_CLASSES = 5  # adjust if you have different labels
TRAIN_DIR = "data/train"
VAL_DIR = "data/val"
SAVE_PATH = "models/trained_model.h5"

def build_model(num_classes):
    base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.4)(x)
    preds = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base.input, outputs=preds)
    return model, base

def main():
    if not os.path.exists(TRAIN_DIR) or not os.path.exists(VAL_DIR):
        print("Please prepare dataset in 'data/train' and 'data/val' with subfolders per class.")
        return

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.12,
        height_shift_range=0.12,
        horizontal_flip=True,
        brightness_range=(0.7,1.3),
        zoom_range=0.15
    )
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH, class_mode='categorical')
    val_gen = val_datagen.flow_from_directory(VAL_DIR, target_size=IMG_SIZE, batch_size=BATCH, class_mode='categorical')

    num_classes = len(train_gen.class_indices)
    model, base = build_model(num_classes)

    # Freeze base
    for layer in base.layers:
        layer.trainable = False

    model.compile(optimizer=Adam(1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())

    # Train head
    model.fit(train_gen, epochs=5, validation_data=val_gen)

    # Unfreeze some layers and fine-tune
    for layer in base.layers[-40:]:
        layer.trainable = True

    model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_gen, epochs=EPOCHS, validation_data=val_gen)

    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    model.save(SAVE_PATH)
    print("Saved model to", SAVE_PATH)

if __name__ == "__main__":
    main()
