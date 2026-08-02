import cv2
import tensorflow as tf
import numpy as np

def process_image(image):
    image = np.array(image)

    image = cv2.resize(image, (256, 256))

    image = tf.cast(image , dtype=tf.float32)

    image = tf.expand_dims(image, axis=0)

    return image
  