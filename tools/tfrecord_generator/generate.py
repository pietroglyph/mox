#!/bin/python3
r"""Convert raw Magic: The Gathering data into a set of ".tfrecord" files.

Example usage:
    python generate.py --logtostderr \
      --data_dir=${DATA_DIR} \
      --output_dir="${OUTPUT_DIR}"
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import hashlib
import io
import json
import os
import glob
import json
import numpy as np

import tensorflow as tf

from PIL import Image, ImageEnhance, ImageDraw
from object_detection.utils import dataset_util

flags = tf.app.flags
tf.flags.DEFINE_string('data_dir', './raw_data/',
                       'Training image directory.')
tf.flags.DEFINE_string('output_dir', '/tmp/', 'Output data directory.')
tf.flags.DEFINE_boolean('show_debug', False, 'Display a debug image with bounding boxes drawn.')

FLAGS = flags.FLAGS
MAX_UNDERFITTING_PIXELS = 80
MAX_COLOR_BALANCE_DELTA = 0.3
MAX_BRIGHTNESS_DELTA = 0.4
TRAINING_SEGMENTS = 2
BACKGROUND_NAME = "background.png"
NUM_EVAL_EXAMPLES = 10

tf.logging.set_verbosity(tf.logging.INFO)

def randomise_image(foreground_bytes, background_path):
    im = Image.open(foreground_bytes)
    bg = Image.open(background_path, mode="r")

    oldsize = im.size

    # Downscale
    im.thumbnail((bg.size[0]-random.randint(0, MAX_UNDERFITTING_PIXELS), bg.size[1]-random.randint(0, MAX_UNDERFITTING_PIXELS)), Image.LANCZOS)

    # Position the image randomly
    margin_x = (bg.size[0]//2)-(im.size[0]//2)
    margin_y = (bg.size[1]//2)-(im.size[1]//2)
    transform_x = (random.choice([-1,1]) * random.randint(0,margin_x))
    transform_y = (random.choice([-1,1]) * random.randint(0,margin_y))
    bg.paste(im, (margin_x + transform_x,  margin_y + transform_y), im)

    # Apply some random image enhancements (color balance, brightness, sharpness, contrast, etc.)
    im = ImageEnhance.Color(im).enhance((random.random() * MAX_COLOR_BALANCE_DELTA * random.choice([-1,1])) + 1)
    im = ImageEnhance.Brightness(im).enhance((random.random() * MAX_BRIGHTNESS_DELTA * random.choice([-1,1])) + 1)

    # bg.size should == (1024, 1024)
    return bg, (transform_x + margin_x, transform_y + margin_y), (im.size[0]/oldsize[0], im.size[1]/oldsize[1]), oldsize

def create_tf_record_from_annotations(in_dir, in_ids, out_path):
    background_png_path = os.path.join(in_dir, BACKGROUND_NAME)
    writer = tf.python_io.TFRecordWriter(out_path)

    for v in in_ids:
        json_path = os.path.join(in_dir, v + ".json")
        with tf.gfile.GFile(json_path, "r") as f:
            json_raw = f.read()
        annotation = json.loads(json_raw)

        filename = v + ".png"
        card_png_path = os.path.join(in_dir, filename)
        print(card_png_path)
        with tf.gfile.GFile(card_png_path, 'rb') as fid:
            encoded_png = fid.read()
        randomised_image, card_transform, card_resize_factor, original_size = randomise_image(io.BytesIO(encoded_png), background_png_path)
        if randomised_image.format != "PNG":
            raise ValueError("Image format not PNG")

        annotation["card"] = {
            "startX": 0,
            "startY": 0,
            "endX": original_size[0],
            "endY": original_size[1],
        }

        startx = []
        starty = []
        endx = []
        endy = []
        category_id = []
        category_name = []

        if FLAGS.show_debug:
            debug_draw = ImageDraw.Draw(randomised_image)

        for k in annotation:
            if k != "uuid":
                # All coordinates are normalized here, too
                c1 = (((annotation[k]["startX"] * card_resize_factor[0]) + card_transform[0]) / randomised_image.size[0], ((annotation[k]["startY"] * card_resize_factor[1]) + card_transform[1]) / randomised_image.size[1])
                c2 = (((annotation[k]["endX"] * card_resize_factor[0]) + card_transform[0]) / randomised_image.size[0], ((annotation[k]["endY"] * card_resize_factor[1]) + card_transform[1]) / randomised_image.size[1])

                if c1 > c2:
                    c1, c2 = c2, c1 # Swap the variables
                    print("Found disordered bounding box; it will be swapped.")

                if FLAGS.show_debug:
                    debug_draw.rectangle([c1[0] * randomised_image.size[0], c1[1] * randomised_image.size[1], c2[0] * randomised_image.size[0], c2[1] * randomised_image.size[1]])

                startx.append(c1[0])
                starty.append(c1[1])
                endx.append(c2[0])
                endy.append(c2[1])

                category_id.append(category_name_to_id(k))
                category_name.append(k.encode("utf8"))

        if FLAGS.show_debug:
            randomised_image.show()
            raw_input("Press any key to show next image.")
            continue

        # This is wildly inefficient... I haven't found a better way to get this
        # into a format TensorFlow wants.
        randomised_image_bytes = io.BytesIO()
        randomised_image.save(randomised_image_bytes, format="PNG")

        example = tf.train.Example(features=tf.train.Features(feature={
            "image/width": dataset_util.int64_feature(randomised_image.size[0]),
            "image/height": dataset_util.int64_feature(randomised_image.size[1]),
            "image/filename": dataset_util.bytes_feature(filename.encode("utf8")),
            "image/source_id": dataset_util.bytes_feature(
                annotation["uuid"].encode("utf8")),
            "image/encoded": dataset_util.bytes_feature(randomised_image_bytes.getvalue()),
            "image/format": dataset_util.bytes_feature(b"png"),
            "image/object/bbox/xmin": dataset_util.float_list_feature(startx),
            "image/object/bbox/xmax": dataset_util.float_list_feature(endx),
            "image/object/bbox/ymin": dataset_util.float_list_feature(starty),
            "image/object/bbox/ymax": dataset_util.float_list_feature(endy),
            "image/object/class/text": dataset_util.bytes_list_feature(category_name),
            "image/object/class/label": dataset_util.int64_list_feature(category_id),
        }))

        writer.write(example.SerializeToString())
    writer.close()



def category_name_to_id(name):
    return {
        "cardName": 1,
        "setSymbol": 2,
        "collectorNumber": 3,
        "typeLine": 4,
        "card": 5
    }[name]

def main(_):
    assert FLAGS.data_dir, '`data_dir` missing.'
    assert FLAGS.output_dir, '`output_dir` missing.'

    if not tf.gfile.IsDirectory(FLAGS.output_dir):
        tf.gfile.MakeDirs(FLAGS.output_dir)
    train_output_path = os.path.join(FLAGS.output_dir, 'mtg_train.record')
    val_output_path = os.path.join(FLAGS.output_dir, 'mtg_val.record')
    # testdev_output_path = os.path.join(FLAGS.output_dir, 'coco_testdev.record')

    files = glob.glob(FLAGS.data_dir+"*.json")
    ids = [[], [], []]
    files_added = 0
    segment = 0
    extra_data_added = False
    for i, v in enumerate(files):
        if files_added > len(files) / TRAINING_SEGMENTS:
            if segment == 0 and not extra_data_added:
                extra_data_added = True
                files_added += (NUM_EVAL_EXAMPLES + 1) - (len(files) / TRAINING_SEGMENTS)
            else:
                files_added = 0
                segment += 1
        ids[segment].append(files[i][len(FLAGS.data_dir):-5]) # strip the .json suffix to get the id
        files_added += 1

    print("Using", len(ids[0]), "images for training, and", len(ids[1]), "for evaluation.")

    create_tf_record_from_annotations(FLAGS.data_dir, ids[0], train_output_path)
    create_tf_record_from_annotations(FLAGS.data_dir, ids[1], val_output_path)
    # create_tf_record_from_annotations(FLAGS.data_dir, ids[2], testdev_output_path)

if __name__ == '__main__':
    tf.app.run()
