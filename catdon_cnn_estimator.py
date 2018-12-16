from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

IMG_SIZE=50


def cnn_model_fn(features,labels,mode):

	#reshape images to 4-D Tensors

	input_layer=tf.reshape(features['x'],[-1,IMG_SIZE,IMG_SIZE,1])

	#apply first conv #1
	#INPUT [50,50]
	#OUTPUT [48,48]

	conv1=tf.layers.conv2d(
		inputs=input_layer,
		filters=32,
		kernel_size=[5,5],
		padding='same',
		activation=tf.nn.relu)

	#apply maxplool #1
	#INPUT [48,48]
	#OUTPUT [24,24]

	pool1=tf.layers.max_pooling2d(inputs=conv1,pool_size=2,strides=2)


	#apply conv #2
	#INPUT [24,24]
	#OUTPUT [22,22]

	conv2=tf.layers.conv2d(
		inputs=pool1,
		filters=64,
		kernel_size=[5,5],
		padding='same',
		activation=tf.nn.relu)

	# apply maxpool #2
	#INPUT [22,22]
	#OUTPUT [11,11]

	pool2=tf.layers.max_pooling2d(inputs=conv2,pool_size=2,strides=2)



	#reshape for densr layer
	pool2_flat=tf.reshape(pool2,[-1,12*12*64])

	#dense layer

	dense=tf.layers.dense(inputs=pool2_flat,units=1024,activation=tf.nn.relu)

	#dropout layer 

	dropout=tf.layers.dropout(inputs=dense,rate=0.4,training=mode==tf.estimator.ModeKeys.TRAIN)


	#logits layer

	#INPUT 1024
	#OUTPUT 2

	logits=tf.layers.dense(inputs=dropout,units=2)

	predictions = {

	'classes':tf.argmax(input=logits,axis=1,name='classes_tensor'),

	'probabilities':tf.nn.softmax(logits,name='softmax_tensor')

	}

	if mode == tf.estimator.ModeKeys.PREDICT:

		return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)


	#calculate the loss

	loss=tf.losses.sparse_softmax_cross_entropy(labels=labels,logits=logits)

	#Configure the Training Op (for TRAIN mode)

	if mode == tf.estimator.ModeKeys.TRAIN:

		optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.001)

		train_op=optimizer.minimize(
			loss=loss,
			global_step=tf.train.get_global_step()
			)

		return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)


	# Add evaluation metrics (for EVAL mode)

	eval_metric_ops = {

	'accuracy':tf.metrics.accuracy(labels=labels,predictions=predictions['classes'])

	}

	return tf.estimator.EstimatorSpec(
    mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


    # END of modeling

def main(unused_argv):

	TRAIN_DATA=np.load('train_data.npy')

	np_features=[i[0] for i in TRAIN_DATA]

	train_data=np.asarray(np_features[0:24000],dtype=np.float32)

	eval_data=np.asarray(np_features[24000:-1],dtype=np.float32)

	np_labels=[i[1] for i in TRAIN_DATA]

	train_labels=np.asarray(np_labels[0:24000])

	eval_labels=np.asarray(np_labels[24000:-1])





	mnist_classifier = tf.estimator.Estimator(
	 	model_fn=cnn_model_fn, 
	 	model_dir=r"/home/eva_01/Desktop/CatDog_estimator/catdog_log")

	tensors_to_log = {'probabilities': 'softmax_tensor'}

	logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=480)


		# Train the model


	train_input_fn = tf.estimator.inputs.numpy_input_fn(
		x={"x": train_data},
		y=train_labels,
		batch_size=50,
		num_epochs=None,
		shuffle=True)

	mnist_classifier.train(
		input_fn=train_input_fn,
		steps=192000,
		hooks=[logging_hook])

	# Evaluate the model and print results

	eval_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": eval_data},y=eval_labels,num_epochs=1,shuffle=False)
		
	eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)

	print(eval_results)


if __name__=='__main__':
	tf.app.run()






