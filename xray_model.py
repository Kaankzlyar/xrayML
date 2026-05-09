import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

train_dir = r"D:\ProjectsD\Odev7\xray-env\datasets\parts_train"
test_dir  = r"D:\ProjectsD\Odev7\xray-env\datasets\parts_test"

IMG_SIZE = 224
BATCH = 16

checkpoint = ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True, verbose=1)
early_stop = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1)

# --- VERİ HAZIRLAMA ---
# VGG16 kendi preprocess_input fonksiyonunu kullanır (ImageNet mean subtraction).
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.15,
    brightness_range=(0.8, 1.2),
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH,
    class_mode='categorical',
    color_mode='rgb',
    shuffle=True,
    subset="training"
)

valid_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH,
    class_mode='categorical',
    color_mode='rgb',
    shuffle=True,
    subset="validation"
)

test_gen = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH,
    class_mode='categorical',
    color_mode='rgb',
    shuffle=False
)

# --- CLASS WEIGHTS (dengesiz sınıflar için) ---
counts = np.bincount(train_gen.classes)
total = counts.sum()
n_classes = len(counts)
class_weight = {i: total / (n_classes * c) for i, c in enumerate(counts)}
print("Class weights:", class_weight)

# --- TRANSFER LEARNING: VGG16 ---
base_model = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
base_model.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(4, activation='softmax')
])

model.summary()

# --- FAZ 1: Sadece head eğitimi ---
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history1 = model.fit(
    train_gen,
    epochs=20,
    validation_data=valid_gen,
    class_weight=class_weight,
    verbose=1,
    callbacks=[checkpoint, early_stop]
)

# --- FAZ 2: VGG16'nın son bloğunu (block5) fine-tune et ---
base_model.trainable = True
for layer in base_model.layers:
    if not layer.name.startswith('block5'):
        layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history2 = model.fit(
    train_gen,
    epochs=20,
    validation_data=valid_gen,
    class_weight=class_weight,
    verbose=1,
    callbacks=[checkpoint, early_stop]
)

model.save('xray_vgg16.h5')
print("Model kaydedildi!")

loss, accuracy = model.evaluate(test_gen)
print(f"Test Accuracy: {accuracy:.4f}")

# --- GRAFİK ---
acc = history1.history['accuracy'] + history2.history['accuracy']
val_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']
loss_h = history1.history['loss'] + history2.history['loss']
val_loss = history1.history['val_loss'] + history2.history['val_loss']
phase_split = len(history1.history['accuracy'])

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(acc, label='Train')
plt.plot(val_acc, label='Valid')
plt.axvline(phase_split - 0.5, color='gray', linestyle='--', label='Fine-tune start')
plt.title('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss_h, label='Train')
plt.plot(val_loss, label='Valid')
plt.axvline(phase_split - 0.5, color='gray', linestyle='--', label='Fine-tune start')
plt.title('Loss')
plt.legend()

plt.savefig('training_results.png')
plt.show()
