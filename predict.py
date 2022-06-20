from process_image import process_image
import argparse

import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import json
import workspace_utils

import logging

logger = tf.get_logger()
logger.setLevel(logging.ERROR)

# Create the predict function
from PIL import Image


def main():
    # code inside main
    in_args = get_input_args()
    model = load_model(in_args.saved_model)
    class_names = load_class_names(in_args.category_names)
    probs, classes = predict(in_args.input_image_path, model, class_names,
                         in_args.top_k)
    flowers_names = [class_names[str(x)] for x in classes]
    
    
def get_input_args():
    parser = argparse.ArgumentParser(description='Image Classifier Command Line Application')
    # code to read all arguments here
    parser.add_argument('input_image_path', action='store', type=str,
                        help='input image path')
    parser.add_argument('saved_model')
    parser.add_argument('--top_k', dest="top_k", type=int)
    parser.add_argument('--category_names', dest="category_names") # action='store', type=str
    args = parser.parse_args()

    # return parsed argument collection
    return args


def load_model(saved_keras_model_filepath):
    '''
          Load a checkpoint from filepath and rebuild model.
    '''
    # Load the Keras model
    reloaded_keras_model = tf.keras.models.load_model(saved_keras_model_filepath,
                                                      custom_objects={'KerasLayer': hub.KerasLayer}, compile=False)
    return reloaded_keras_model
    # reloaded_keras_model.summary()


def load_class_names(label_map):
    if label_map is not None:
        with open(label_map, 'r') as f:
            return json.load(f)
    else:
        print('No label_map was provided!')


    
def predict(image_path, model, class_names, top_k=5):
    img = Image.open(image_path)
    np_img = np.asarray(img)
    processed_np_img = process_image(np_img)
    np_img_expanded = np.expand_dims(processed_np_img, axis=0)
    probs = model.predict(np_img_expanded)
    probs0 = probs[0].tolist()
    topk_probs, idx = tf.math.top_k(probs0, k=top_k)
    np_probs = topk_probs.numpy().tolist()
    classes = (idx + 1).numpy().tolist()
    return np_probs, classes
    

  
# image_path = './test_images/hard-leaved_pocket_orchid.jpg'
# probs, classes = predict(image_path, model, 5)
# print(probs)
# print(classes)
# print('\n')
# [class_names[str(classes[i])] for i in range(len(classes))]


if __name__ == "__main__":
    main()


