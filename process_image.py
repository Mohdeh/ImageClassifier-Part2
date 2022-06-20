import tensorflow as tf
# Create the process_image function

image_size = 224

def process_image(image):
    # image = tf.cast(image, tf.float32)
    tf_image = tf.convert_to_tensor(image, dtype=tf.float32)
    tf_image = tf.image.resize(tf_image, (image_size, image_size))
    tf_image /= 255
    np_image = tf_image.numpy()
    return np_image
