#!/usr/bin/env python
import argparse
import numpy as np
import tensorflow as tf
import os.path as osp

import models
import dataset
#from models import vgg16


def display_results(image_paths, probs):
    '''Displays the classification results given the class probability for each image'''
    # Get a list of ImageNet class labels
    with open('imagenet-classes.txt', 'rb') as infile:
    	lines = infile.readlines()
    	class_labels = []
    	for line in lines:
    		class_labels.append(str(line).strip())
    # Pick the class with the highest confidence for each image
    class_indices = np.argmax(probs, axis=1)
    #print(class_indices)
    # Display the results
    print('\n{:20} {:30} {}'.format('Image', 'Classified As', 'Confidence'))
    print('-' * 70)
    for img_idx, image_path in enumerate(image_paths):
        img_name = osp.basename(image_path)
        #print(len(list(class_labels)))
        class_name = class_labels[class_indices[img_idx]]
        confidence = round(probs[img_idx][class_indices[img_idx]] * 100, 2)
        print('{:20} {:30} {} %'.format(img_name, class_name, confidence))


def classify(model_data_path, image_paths):
    '''Classify the given images using GoogleNet.'''

    # Get the data specifications for the GoogleNet model
    #spec = models.get_data_spec(model_class=models.GoogleNet)
    spec = models.get_data_spec(model_class=models.VGG_ILSVRC_16_layer)

    # Create a placeholder for the input image
    input_node = tf.placeholder(tf.float32,
                                shape=(None, spec.crop_size, spec.crop_size, spec.channels))

    # Construct the network
    #net = models.GoogleNet({'data': input_node})
    net = models.VGG_ILSVRC_16_layer({'data': input_node})


    def load_image(image_path, data_spec):
        # Read the file
        file_data = tf.read_file(image_path)
        # Decode the image data
        img = tf.image.decode_jpeg(file_data, channels=data_spec.channels)
        if data_spec.expects_bgr:
            # Convert from RGB channel ordering to BGR
            # This matches, for instance, how OpenCV orders the channels.
            img = tf.reverse(img, [2, 1, 0])
        return img

    # Create an image producer (loads and processes images in parallel)
    image_producer = dataset.ImageProducer(image_paths=image_paths, data_spec=spec)

    with tf.Session() as sesh:
        # Start the image processing workers
        #coordinator = tf.train.Coordinator()
        #threads = image_producer.start(session=sesh, coordinator=coordinator)

        # Load the converted parameters
        print('Loading the model')
        net.load(model_data_path, sesh)

        # Load the input image
        print('Loading the images')
        is_jpeg = True
        probs = []
        for image_path in image_paths:
        	input_image = load_image(image_path, image_producer.data_spec)
        	processed_img = dataset.process_image(img=input_image,
                                      scale=image_producer.data_spec.scale_size,
                                      isotropic=image_producer.data_spec.isotropic,
                                      crop=image_producer.data_spec.crop_size,
                                      mean=image_producer.data_spec.mean)
        	#print('Classifying')
        	prob = sesh.run(net.get_output(), feed_dict={input_node: np.reshape(processed_img.eval(), (1, 224, 224, 3))})
        	probs.extend(prob)
        #indices, input_images = image_producer.get(sesh)
        indices = range(len(image_paths))
        # Perform a forward pass through the network to get the class probabilities
        
        display_results([image_paths[i] for i in indices], probs)

        # Stop the worker threads
        #coordinator.request_stop()
        #coordinator.join(threads, stop_grace_period_secs=2)

def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', help='Converted parameters for the GoogleNet model')
    parser.add_argument('image_paths', nargs='+', help='One or more images to classify')
    args = parser.parse_args()

    # Classify the image
    classify(args.model_path, args.image_paths)


if __name__ == '__main__':
    main()
