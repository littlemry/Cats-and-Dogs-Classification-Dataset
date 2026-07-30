import tensorflow as tf
import numpy as np

MODEL_PATH = "cat_dog_model.keras"
IMAGE_PATH = "OIP.jpeg"

model = tf.keras.models.load_model(MODEL_PATH)


def predict_image(image_path: str = IMAGE_PATH) -> str:
    img = tf.keras.preprocessing.image.load_img(
        image_path,
        target_size=(128, 128),
    )

    img = tf.keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0

    prediction = model.predict(img, verbose=0)

    return "Dog" if prediction[0][0] > 0.5 else "Cat"


def app():
    return predict_image()


if __name__ == "__main__":
    print(predict_image())