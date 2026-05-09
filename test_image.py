from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import matplotlib.pyplot as plt

# --- MODELİ YÜKLE ---
model = load_model('best_model.h5')

# --- SINIF İSİMLERİ ---
classes = ['bilek', 'dirsek', 'el', 'omuz']

# --- TEST ETMEK İSTEDİĞİN GÖRÜNTÜ ---
image_path = r"D:\ProjectsD\Odev7\xray-env\datasets\parts_test\bilek\study1_negative_image1.png"  

# --- GÖRÜNTÜYÜ HAZIRLA ---
image = load_img(image_path, target_size=(128, 128), color_mode="rgb")
image_array = img_to_array(image) / 255.0
image_array = np.expand_dims(image_array, axis=0)

# --- TAHMİN YAP ---
predictions = model.predict(image_array)

predicted_class = classes[predictions.argmax()]
confidence = predictions.max() * 100

# --- SONUCU GÖSTER ---
plt.figure(figsize=(6, 6))
plt.imshow(load_img(image_path), cmap='gray')
plt.title(f"Tahmin: {predicted_class}\nGüven: %{confidence:.1f}")
plt.axis('off')
plt.show()

# --- TÜM SINIF OLASILIKLARINI GÖSTER ---
print("\nTüm tahminler:")
for i, cls in enumerate(classes):
    print(f"  {cls}: %{predictions[0][i]*100:.1f}")