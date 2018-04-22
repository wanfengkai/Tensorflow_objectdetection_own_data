# Tensorflow\_object\_detection\_on\_own\_data

Following this README you will be able to train and evaluate all the object detection models with your own dataset.


## tensorflow preparation
Normally, we install tensorflow via ```pip install tensorflow-gpu/tensorflow```. This will give us all the tensorflow packages but the ```models/``` folder is absent which contains all the model APIs that we will use.

To prepare the tensorflow to run Object detection on your local machine, you should follow the steps below:

* Install Tensorflow: on my AWS AMI server which has Ubuntu 16 system, we can install Tensorflow by: ```pip install tensorflow-gpu```;

* Install other packages that are required [details here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md): 

		sudo apt-get install protobuf-compiler python-pil python-lxml python-tk
		sudo pip install Cython
		sudo pip install jupyter
		sudo pip install matplotlib

* Get the ```models/``` folder by ```git clone https://github.com/tensorflow/models.git```
	* Note that there are some bugs in the newest offical repository(_2018.04.18_), to avoid that you can simply check to the formal commit and only change one line in a python script:
		*  Get to the cloned ```models/``` folder ```cd models```
		*  Get the older commit by```git checkout -b before-object-detection-changes 2913cb24ecb16ba955006b41072adce45c5a0f62```
		*  Modify the code by: ```vim models/research/object_detection/utils/learning_schedules.py```.(You might come across permission problem,just ```sudo chmod 777 models/research/object_detection/utils/learning_schedules.py``` to change the permission and then change the code using vim or whatever editor you want to use. )
		
				The line need to be changed is in line 153 :
				You need to change from: 
				tf.constant(range(num_boundaries), dtype=tf.int32),
				to:
				tf.constant(list(range(num_boundaries)), dtype=tf.int32),
* Copy or move the ```models/``` to the tensorflow folder.
	* First thing to do---find the tensorflow: ```pip show tensorflow```.  You will get following logs:(The ***Location*** indicate the path to tensorflow)
	
			Name: tensorflow
			Version: 1.5.0
			Summary: TensorFlow helps the tensors flow
			Home-page: https://www.tensorflow.org/
			Author: Google Inc.
			Author-email: opensource@google.com
			License: Apache 2.0
			Location: /home/ubuntu/anaconda3/lib/python3.6/site-packages
			Requires: absl-py, six, numpy, wheel, protobuf, tensorflow-tensorboard 
	* Copy the ```models/``` to the tensorflow folder by:
	```sudo cp -r models/ /home/ubuntu/anaconda3/lib/python3.6/site-packages/tensorflow/ ```

* COCO API installation:

		git clone https://github.com/cocodataset/cocoapi.git
		cd cocoapi/PythonAPI
		make
		cp -r pycocotools <path_to_tensorflow>/models/research/
* Protobuf Compilation:(***cd to ```tensorflow/models/research/```***)

		# From tensorflow/models/research/
		protoc object_detection/protos/*.proto --python_out=.
* Add Libraries to PYTHONPATH:

		# From tensorflow/models/research/
		export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
		
* Testing the Installation:

		python object_detection/builders/model_builder_test.py
	
If everything goes well then you are done with Tensorflow preparation.

## Generate your own dataset.

This is the key part, you should create ```tfrecord``` file with your dataset:

* Get this repository: ```git clone https://github.com/wanfengkai/Tensorflow_objectdetection_own_data.git```
* Put your data (images and txt files) to ```./datademo``` or anywhere you want to but just make sure you change the ```input_path``` default value in ```data_converter.py```
* Same principle for the ```output_path```
* In the code ```data_converter.py``` make sure your name patterns for dataset is same as the line after ***TODO*** . Make further modification if you need to.

* Change the _***dict***_ according to your dataset classes.
* Change the _***ORIGINAL\_WIDTH***_  and _***ORIGINAL\_HEIGHT***_ according to the original resolution if your annotation's coordinate is based on the original size/resolution of your images.

* Simply create your tfrecord by ```python data_converter.py```.
	* Note that if your dataset is large, try to generate them in multiple files instead of one, this will not cause error in future training. This can be done by Comment and uncomment some lines as it mentioned clearly in code.

* Generate LabelMap manually for future use. You can simply follow the pattern in ```./tfrecord_folder/traffisign.pbtxt```


##Get pretrained model according to your favor in model zoo
* You can get any model you want train or eval in [model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md).
* In this case, we take [faster\_rcnn\_inception\_resnet\_v2\_atrous\_coco](http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28.tar.gz)
* Download the model:```wget http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28.tar.gz```
* Unzip the model:```tar -xzvf faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28.tar.gz```
* Change the ```pipeline.config``` in the unziped folder.
	* The things that you need to change are the ones ***```PATH_TO_BE_CONFIGURED ```***, according to their indication, you should set the path for them. For example the pathes for output models ,input tfrecord and labelmap.


## Train or eval your model 

* Firstly change your current working directory  ```tensorflow/models/research/``` then:

			# From tensorflow/models/research/
			export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
* Claim some global variables beforehand: 

		PATH_TO_YOUR_PIPELINE_CONFIG='/home/ubuntu/faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28/pipeline.config'
		PATH_TO_TRAIN_DIR='/home/ubuntu/trained_faster'
* Train your model simply:

		# From the tensorflow/models/research/ directory
		python object_detection/train.py \
		    --logtostderr \
		    --pipeline_config_path=${PATH_TO_YOUR_PIPELINE_CONFIG} \
		    --train_dir=${PATH_TO_TRAIN_DIR}
	
* Eval your model:

		# From the tensorflow/models/research/ directory
		PATH_TO_EVAL_DIR='the directory in which evaluation events will be saved'
		python object_detection/eval.py \
		    --logtostderr \
		    --pipeline_config_path=${PATH_TO_YOUR_PIPELINE_CONFIG} \
		    --checkpoint_dir=${PATH_TO_TRAIN_DIR} \
		    --eval_dir=${PATH_TO_EVAL_DIR}
* TENSORBOARD:

		PATH_TO_MODEL_DIRECTORY='directory that contains the train and eval directories.'
		tensorboard --logdir=${PATH_TO_MODEL_DIRECTORY}

