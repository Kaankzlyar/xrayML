from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

model = load_model('xray_model.h5')

test_dir = r"D:\ProjectsD\Odev7\xray-env\datasets\parts_test"

test_datagen = ImageDataGenerator(rescale=1./255)
test_gen = test_datagen.flow_from_directory(
    test_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    color_mode='rgb',
    shuffle=False
)

# --- GENEL BAŞARI ---
loss, accuracy = model.evaluate(test_gen)
print(f"\nTest Accuracy: %{accuracy*100:.1f}")
print(f"Test Loss: {loss:.4f}")

# --- CONFUSION MATRIX ---
test_gen.reset()
preds = model.predict(test_gen)
y_pred = np.argmax(preds, axis=1)
y_true = test_gen.classes
classes = list(test_gen.class_indices.keys())

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', 
            xticklabels=classes, 
            yticklabels=classes, 
            cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('Gerçek')
plt.xlabel('Tahmin')
plt.savefig('confusion_matrix.png')
plt.show()

# --- DETAYLI RAPOR ---
print("\nDetaylı Rapor:")
print(classification_report(y_true, y_pred, target_names=classes))