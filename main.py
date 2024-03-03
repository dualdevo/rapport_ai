import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import os
import matplotlib
matplotlib.use('tkAgg')
import matplotlib.pyplot as plt
import os

train_folder = "E:\\pimg"
test_folder = "E:\\pimgtest"

layers_folder = 'layers'
if not os.path.exists(layers_folder):
    os.makedirs(layers_folder)


def load_images(folder_path, nb_digit, nb_version):
    images = []
    labels = []
    for digit in range(nb_digit):
        for version in range(nb_version):
            image_path = os.path.join(folder_path, f"{digit}_{version}.bmp")
            image = tf.keras.preprocessing.image.load_img(image_path, color_mode='grayscale', target_size=(28, 28))
            image = tf.keras.preprocessing.image.img_to_array(image)
            image /= 255.0
            images.append(image)
            labels.append(digit)
    return np.array(images), np.array(labels)

x, y = load_images(train_folder, 10, 20)

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.15, random_state=42)

x_test, y_test = load_images(test_folder, 10, 2)  # Utilisation de 2 versions pour chaque chiffre

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=20, batch_size=1, validation_data=(x_val, y_val))

eval_loss, eval_acc = model.evaluate(x_test, y_test, batch_size=1)
print('Accuracy: ', eval_acc*100)

predictions = model.predict(x_test)

plt.figure(figsize=(10, 10))

for layer in model.layers:
    weights = layer.get_weights()
    if len(weights) != 0:
        w, b = weights
        np.savetxt('layers/layer_weight_' + layer.name + '.txt', w)
        np.savetxt('layers/layer_bias_' + layer.name + '.txt', b)


for i in range(len(x_test)):
    plt.subplot(5, 4, i+1)  # 5 rows, 4 columns for 20 images
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    plt.title(f"Label réel: {y_test[i]}, Prédiction: {np.argmax(predictions[i])}")
    plt.axis('off')
plt.show()