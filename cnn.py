""" CNN inference on images

Parts of the code are from the tutorial below:
https://github.com/tensorflow/models/blob/master/research/slim/slim_walkthrough.ipynb
"""

import argparse
import numpy as np
import os
import tensorflow as tf
from PIL import Image
import timeit

try:
    import urllib2 as urllib
except ImportError:
    import urllib.request as urllib

import sys
sys.path.insert(0, '../models/research/slim/')
from datasets import imagenet, dataset_utils
from nets import inception
from preprocessing import inception_preprocessing

from tensorflow.contrib import slim

image_size = inception.inception_v4.default_image_size
names = imagenet.create_readable_names_for_imagenet_labels()


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--checkpoints_dir', default='../checkpoints',
                        help='Directory of checkpoints; contains *.ckpt')
    parser.add_argument('-i', '--image_dir', default='/Volumes/Kritkorn/imagenet',
                        help='Directory containing subdirectories of images (wnid)')
    parser.add_argument('-o', '--output_dir', default='/Volumes/Kritkorn/results',
                        help='Directory to output inference results')
    parser.add_argument('-s', '--synset_list', default='available_hop2.txt',
                        help='File containing list of wnids to process')
    parser.add_argument('-m', '--model_name', default='inception_v4',
                        help='CNN model to use, e.g. inception_v4')
    parser.add_argument('-d', '--debug', dest='debug', action='store_true',
                        help='Use debug mode, printing more info')
    parser.add_argument('-w', '--overwrite', dest='overwrite', action='store_true',
                        help='overwrite inference results')
    # TODO: support other models; could also use frozen pb graph instead
    return parser


def download_and_extract_pretrained_model(args):
    """Download and extract pre-trained model from tensorflow website."""
    if tf.gfile.Exists(os.path.join(args.checkpoints_dir, args.model_name + '.ckpt')):
        return
    if not tf.gfile.Exists(args.checkpoints_dir):
        tf.gfile.MakeDirs(args.checkpoints_dir)

    url_base = "http://download.tensorflow.org/models/"
    model_tarballs = {
        "inception_v1" : "inception_v1_2016_08_28.tar.gz",
        "inception_v2" : "inception_v2_2016_08_28.tar.gz",
        "inception_v3" : "inception_v3_2016_08_28.tar.gz",
        "inception_v4" : "inception_v4_2016_09_09.tar.gz",
    }
    url = url_base + model_tarballs[args.model_name]
    dataset_utils.download_and_uncompress_tarball(url, args.checkpoints_dir)


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    image_np = np.array(image.getdata()).reshape((im_height, im_width, -1)).astype(np.uint8)
    if image_np.shape[2] == 1:
        image_np = np.repeat(image_np, 3, axis=2)
        # print image_np.shape
        # image_np = image_np[:, :, None] * np.ones(3, dtype=int)[None, None, :]
    assert image_np.shape[2] == 3
    return image_np


def run_inference(args):
    with open(args.synset_list, 'r') as f:
        synsets_to_process = map(lambda x: x.strip(), f.readlines())

    with tf.Graph().as_default():
        image_shape = (None, None, 3)
        image_placeholder = tf.placeholder(tf.uint8, shape=image_shape, name='image_placeholder')
        processed_image = inception_preprocessing.preprocess_image(image_placeholder, image_size, image_size, is_training=False)
        processed_images  = tf.expand_dims(processed_image, 0)
        
        # Create the model, use the default arg scope to configure the batch norm parameters.
        with slim.arg_scope(inception.inception_v4_arg_scope()):
            logits, _ = inception.inception_v4(processed_images, num_classes=1001, is_training=False)
        probabilities_placeholder = tf.nn.softmax(logits)
        
        init_fn = slim.assign_from_checkpoint_fn(
            os.path.join(args.checkpoints_dir, args.model_name + '.ckpt'),
            slim.get_model_variables('InceptionV4'))
        
        with tf.Session() as sess:
            init_fn(sess)
            existing_synsets_to_process = filter(lambda x: x in synsets_to_process, os.listdir(args.image_dir))
            path_to_synset_dirs = filter(os.path.isdir, 
                map(lambda x: os.path.join(args.image_dir, x), existing_synsets_to_process))

            for i, synset_dir in enumerate(path_to_synset_dirs):
                start_time = timeit.default_timer()
                count_images = 0
                image_names = filter(lambda x : not x.startswith('.'), os.listdir(synset_dir))
                for j, image_name in enumerate(image_names):
                    print "Processing %s (count %d of %d) in synset %s (count %d of %d)"\
                        % (image_name, j+1, len(image_names), synset_dir[-9:], i+1, len(path_to_synset_dirs))
                    filename = os.path.splitext(image_name)[0]
                    output_name = os.path.join(args.output_dir, filename + '.txt')

                    if not args.overwrite and tf.gfile.Exists(output_name):
                        print "Skipped"
                        continue
                    if not tf.gfile.Exists(args.output_dir):
                        tf.gfile.MakeDirs(args.output_dir)

                    io_start_time = timeit.default_timer()
                    try:
                        image = Image.open(os.path.join(synset_dir, image_name))
                        image_np = load_image_into_numpy_array(image)
                    except Exception as e:
                        with open("errors.txt", "a") as f:
                            f.write(image_name + "\n")
                            f.write(str(e) + "\n")
                        continue
                    image_np_expanded = np.expand_dims(image_np, axis=0)
                    io_end_time = timeit.default_timer()

                    probabilities = sess.run(probabilities_placeholder, feed_dict={image_placeholder: image_np})
                    probabilities = probabilities[0, 0:]
                    inference_end_time = timeit.default_timer()

                    np.savetxt(output_name, probabilities)
                    output_end_time = timeit.default_timer()

                    print "Finished writing output to file"
                    if args.debug:
                        print "Input time: %.3f s" % (io_end_time - io_start_time)
                        print "Inference time: %.3f s" % (inference_end_time - io_end_time)
                        print "Output time: %.3f s" % (output_end_time - inference_end_time)
                    count_images += 1

                    # sorted_inds = [ind[0] for ind in sorted(enumerate(-probabilities), key=lambda x:x[1])]
                    # top_count = 5
                    # print('\nTop %d for %s' % (top_count, image_name))
                    # for k in range(top_count):
                    #     index = sorted_inds[k]
                    #     print('Probability %0.2f%% => [%s]' % (probabilities[index] * 100, names[index]))
                end_time = timeit.default_timer()
                total_time = end_time - start_time
                if count_images != 0:
                    average_time = total_time / count_images
                else:
                    average_time = 0
                print "Total time: %.3f s, Average time: %.3f s, Count: %d"\
                    % (total_time, average_time, count_images)


if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()
    download_and_extract_pretrained_model(args)
    run_inference(args)