import tensorflow as tf
import os
import numpy as np
import skimage.io
from tensorflow.models.research.object_detection.utils import dataset_util


flags = tf.app.flags
flags.DEFINE_string('output_path', './tfrecord_folder', 'Path to output TFRecord')
flags.DEFINE_string('input_path', './datademo', 'Path to input files including images and txt annotation files')
FLAGS = flags.FLAGS

ORIGINAL_WIDTH=1628
ORIGINAL_HEIGHT=1236
mydict={"speed_limit" : 1,
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


def create_tf_example(image_path,image_name,height,width,sign_type_num,x_min,y_min,x_max,y_max):
    """Creates a tf.Example proto from image.

    Args:
    example: The jpg encoded data of the cat image.

    Returns:
    example: The created tf.Example.
    """
    
    # TODO(user): Populate the following variables from your example.

    filename = image_name # Filename of the image. Empty if image is not from file
    encoded_image_data =image_data = tf.gfile.FastGFile(image_path, 'rb').read()
    image_format = b'jpeg' # b'jpeg' or b'png'

    xmins = [x_min/ORIGINAL_WIDTH] # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = [x_max/ORIGINAL_WIDTH]# List of normalized right x coordinates in bounding box
             # (1 per box)
    ymins = [y_min/ORIGINAL_HEIGHT] # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = [y_max/ORIGINAL_HEIGHT] # List of normalized bottom y coordinates in bounding box
             # (1 per box)
    classes = [sign_type_num] # List of integer class id of bounding box (1 per box)
    classes_text =list(mydict.keys())[list(mydict.values()).index(sign_type_num)]
    print(classes_text)# List of string class name of bounding box (1 per box)

    tf_example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(filename.encode()),
      'image/source_id': dataset_util.bytes_feature(filename.encode()),
      'image/encoded': dataset_util.bytes_feature(encoded_image_data),
      'image/format': dataset_util.bytes_feature(image_format),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
      'image/object/class/text': dataset_util.bytes_list_feature([classes_text.encode()]),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def read_onlyUsefulinfo_from_original_txt(input_path):
    useful_informations_list=[]

    file_list = os.listdir(input_path)
    for file in file_list:
        if file.endswith('txt'):
            print(file)
            with open(os.path.join(FLAGS.output_path,file.split('.')[0]+'_simple.txt'),'w+') as new_txt:
                with open(os.path.join(input_path,file),'r+') as txt_file:
                    head=txt_file.readline()
                    # head[0],head[1],head[-4],head[-3],head[4],head[5]
                    # useful item: frameNumber(0),signType(1),ulx(-4),uly(-3),lrx(4),lry(5)
                    print(head)
                    for line in txt_file:

                        line=line.split('_')
                        new_txt.write('{},{},{},{},{},{}\n'.format(line[0],line[1],line[-4],line[-3],line[4],line[5]))





def main(_):
    # TODO:if you want to create multiple tfrecord file.Comment this following line.
    train_writer = tf.python_io.TFRecordWriter(os.path.join(FLAGS.output_path,'train.record'))
    # comment this when you done this once
    read_onlyUsefulinfo_from_original_txt(input_path=FLAGS.input_path)
    for file in os.listdir(FLAGS.output_path):
        if '_simple.txt' in file:
            prefix_num_in_txt = file.split('_simple')[0]
            #TODO:Uncomment this if you want to create multiple tfrecord file.Comment this following line.
            # train_writer = tf.python_io.TFRecordWriter(os.path.join(FLAGS.output_path, 'train_{}.record'.format(prefix_num_in_txt)))
            with open(os.path.join(FLAGS.output_path,file),'r') as reader:
                for line in reader:
                    line=line.split(',')
                    frame_num=line[0]
                    print(frame_num)
                    sign_type_num=int(line[1])
                    x_min = int(line[2])
                    y_min = int(line[-1])
                    x_max = int(line[-2])
                    y_max = int(line[-3])
                    # TODO: This  should be modified according to your own naming pattern in the original dataset.
                    image_name = prefix_num_in_txt+'_'+frame_num+'.jpg'
                    image_path=os.path.join(FLAGS.input_path,image_name)
                    image=skimage.io.imread(image_path)
                    height=image.shape[0]
                    width=image.shape[1]
                    tf_example = create_tf_example(image_path,image_name,height,width,sign_type_num,x_min,y_min,x_max,y_max)
                    train_writer.write(tf_example.SerializeToString())
            # TODO:Uncomment this if you want to create multiple tfrecord file.Comment this following line.
            # train_writer.close()
    train_writer.close()


if __name__ == '__main__':
    tf.app.run()