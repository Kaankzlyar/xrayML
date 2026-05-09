import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy

# --- KLASÖR YOLLARI ---
train_dir = r"D:\ProjectsD\Odev7\xray-env\datasets\parts_train"
test_dir  = r"D:\ProjectsD\Odev7\xray-env\datasets\parts_test"

# --- VERİ HAZIRLAMA ---
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_gen = datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),
    batch_size=64,
    class_mode='categorical',
    color_mode='grayscale',
    shuffle=True,
    subset="training"
)

valid_gen = datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),
    batch_size=64,
    class_mode='categorical',
    color_mode='grayscale',
    shuffle=True,
    subset="validation"
)

test_gen = datagen.flow_from_directory(
    test_dir,
    target_size=(128, 128),
    batch_size=64,
    class_mode='categorical',
    color_mode='grayscale'
)

print("Sınıflar:", train_gen.class_indices)
print("Train:", train_gen.samples, "görüntü")
print("Valid:", valid_gen.samples, "görüntü")
print("Test:", test_gen.samples, "görüntü")

# --- MODEL ---
model = Sequential()
model.add(Conv2D(32, (5,5), padding='same', activation='relu', input_shape=(128, 128, 1)))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(128, (3,3), activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(4, activation='softmax'))

model.summary()

# --- DERLEME ---
model.compile(
    loss=categorical_crossentropy,
    optimizer=Adam(learning_rate=0.0001),
    metrics=['accuracy']
)

# --- EĞİTİM ---
history = model.fit(
    train_gen,
    epochs=15,
    verbose=1,
    validation_data=valid_gen
)

# --- KAYDET ---
model.save('xray_model.h5')
print("Model kaydedildi!")

# --- TEST ---
loss, accuracy = model.evaluate(test_gen)
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test Loss: {loss:.4f}")

# --- GRAFİK ---
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Valid')
plt.title('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Valid')
plt.title('Loss')
plt.legend()

plt.savefig('training_results.png')
plt.show()
print("Grafik kaydedildi!")