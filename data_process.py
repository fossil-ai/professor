import os
import json
from PIL import Image
import tensorflow as tf
import io


'''
Faisal Mohammad

This effort is for our final project in CMSC 673.

The following script is used to properly process
the annotated data from the labelme software, and
serialize it into TFRecord files for training.

This code is entirely custom and written to 
accomadate the TFRecordDataset API provided by TF.

'''


class Example:

    def __init__(self, img_filename, width, height, annotationMap):
        self.img_filename = img_filename
        self.caption = annotationMap['label']
        self.bbox = annotationMap['points']
        self.label = 0
        self.width = width
        self.height = height


def process(data_path):
    examples = []
    for i, slide_dir in enumerate(os.listdir(data_path)):
        print("Parsing through " + slide_dir)
        for j, filename in enumerate(os.listdir(data_path + "/" + slide_dir)):
            print(filename)
            if filename.endswith(".json"):
                with open(data_path + "/" + slide_dir + "/" + filename) as out_file:
                    json_data = json.load(out_file)
                    annotations = json_data['shapes']
                    img_filename = data_path + "/" + slide_dir + \
                        "/" + filename.split(".")[0] + ".jpg"
                    image = Image.open(img_filename)
                    width, height = image.size
                    for ann in annotations:
                        example = Example(img_filename, width, height, ann)
                        examples.append(example)
    return examples


def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
    """Wrapper for inserting float features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        # BytesList won't unpack a string from an EagerTensor.
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def tf_example(example):

    with tf.io.gfile.GFile(example.img_filename, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)

    bbox = example.bbox
    xmin = bbox[0][0] / example.width
    xmax = bbox[1][0] / example.width
    ymin = bbox[0][1] / example.height
    ymax = bbox[1][1] / example.height

    feature = {
        'height': _int64_feature(example.height),
        'width': _int64_feature(example.width),
        'depth': _int64_feature(3),
        'label': _int64_feature(example.label),
        'image_raw': _bytes_feature(encoded_jpg),
        'caption': _bytes_feature(example.caption.encode('utf8')),
        'xmin': _float_feature(xmin),
        'xmax': _float_feature(xmax),
        'ymin': _float_feature(ymin),
        'ymax': _float_feature(ymax),
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))


def tf_string_example(example):
    feature = {
        'caption': _bytes_feature(example.caption.encode('utf8')),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def create_tf_records(examples):
    with tf.io.TFRecordWriter("data/tf_records/all_data.record") as writer:
        for i in range(len(examples)):
            example = examples[i]
            print("Writing " + example.img_filename + " to TFRecord")
            example = tf_example(example)
            writer.write(example.SerializeToString())


def create_string_tf_records(examples):
    with tf.io.TFRecordWriter("data/tf_records/captions.record") as writer:
        for i in range(len(examples)):
            example = examples[i]
            example = tf_string_example(example)
            writer.write(example.SerializeToString())


examples = process("data/images")
create_tf_records(examples)
create_string_tf_records(examples)