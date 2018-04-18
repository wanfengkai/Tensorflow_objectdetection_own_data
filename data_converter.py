import tensorflow as tf
import os
# from tensorflow.models.research.object_detection.utils import dataset_util


flags = tf.app.flags
flags.DEFINE_string('output_path', './tfrecord_folder', 'Path to output TFRecord')
flags.DEFINE_string('input_path', './datademo', 'Path to input files including images and txt annotation files')
FLAGS = flags.FLAGS


def dictionary_for_trafficsign():
    dict={"speed_limit" : 1,
          "goods_vehicles": 2,
          "no_overtaking": 3,
          "no_stopping": 4,
          "no_parking": 5,
          "stop": 6,
          "bicycle": 7,
          "hump": 8,
          "no_left": 9,
          "no_right": 10,
          "priority_to": 11,
          "no_entry": 12,
          "yield": 13,
          "parking": 14}
    return dict


def create_tf_example(example):
    # TODO(user): Populate the following variables from your example.
    height = None # Image height
    width = None # Image width
    filename = None # Filename of the image. Empty if image is not from file
    encoded_image_data = None # Encoded image bytes
    image_format = None # b'jpeg' or b'png'

    xmins = [] # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = [] # List of normalized right x coordinates in bounding box
             # (1 per box)
    ymins = [] # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = [] # List of normalized bottom y coordinates in bounding box
             # (1 per box)
    classes_text = [] # List of string class name of bounding box (1 per box)
    classes = [] # List of integer class id of bounding box (1 per box)

    tf_example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(filename),
      'image/source_id': dataset_util.bytes_feature(filename),
      'image/encoded': dataset_util.bytes_feature(encoded_image_data),
      'image/format': dataset_util.bytes_feature(image_format),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def read_onlyUsefulinfo_from_txt(input_path):
    useful_informations_list=[]

    file_list = os.listdir(input_path)
    for file in file_list:
        if file.endswith('txt'):
            print(file)
            with open(os.path.join(input_path,file),'r+') as txt_file:
                head=txt_file.readline()
                print(head)
                useful_informations_list.append(zip(head[0],head[1],head[-4],head[-3],head[4],head[5]))
                # useful item: frameNumber(0),signType(1),ulx(-4),uly(-3),lrx(4),lry(5)
                print(useful_informations_list)



def main(_):
    dict_for_traffic=dictionary_for_trafficsign()

    # TODO(user):Read bounding box information from txt file.

    useful_informations_list = read_onlyUsefulinfo_from_txt(input_path=FLAGS.input_path)

    # writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
    #
    # # TODO(user): Write code to read in your dataset to examples variable
    #
    # for example in examples:
    #   tf_example = create_tf_example(example)
    #   writer.write(tf_example.SerializeToString())
    #
    # writer.close()


if __name__ == '__main__':
    tf.app.run()