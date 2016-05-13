

import tensorflow as tf 
import numpy as np
import math


# Hyper Parameters
LAYER1_SIZE = 200
LAYER2_SIZE = 100
LEARNING_RATE = 0.0001
TAO = 0.001
BATCH_SIZE = 64

class ActorNetwork:
	"""docstring for ActorNetwork"""
	def __init__(self,state_size,action_size):

		self.graph = tf.Graph()
		with self.graph.as_default():
			self.sess = tf.InteractiveSession()

			# create actor network
			self.state_input,\
			self.W1,\
			self.b1,\
			self.W2,\
			self.b2,\
			self.W3,\
			self.b3,\
			self.action_output = self.create_network(state_size,action_size)

			# create target actor network
			self.target_state_input,\
			self.target_W1,\
			self.target_b1,\
			self.target_W2,\
			self.target_b2,\
			self.target_W3,\
			self.target_b3,\
			self.target_action_output = self.create_network(state_size,action_size)

			# define training rules

			self.q_gradient_input = tf.placeholder("float",[None,action_size])
			self.parameters = [self.W1,self.b1,self.W2,self.b2,self.W3,self.b3]
			self.parameters_gradients = tf.gradients(self.action_output,self.parameters,-self.q_gradient_input/BATCH_SIZE)
			self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(zip(self.parameters_gradients,self.parameters))

			self.sess.run(tf.initialize_all_variables())

			self.saver = tf.train.Saver()
			checkpoint = tf.train.get_checkpoint_state("saved_networks")
			if checkpoint and checkpoint.model_checkpoint_path:
				self.saver.restore(self.session, checkpoint.model_checkpoint_path)
				print "Successfully loaded:", checkpoint.model_checkpoint_path
			else:
				print "Could not find old network weights"

			# copy target parameters 
			self.sess.run([
				self.target_W1.assign(self.W1),
				self.target_b1.assign(self.b1),
				self.target_W2.assign(self.W2),
				self.target_b2.assign(self.b2),
				self.target_W3.assign(self.W3),
				self.target_b3.assign(self.b3)
			])



			



	def create_network(self,state_size,action_size):
		layer1_size = LAYER1_SIZE
		layer2_size = LAYER2_SIZE

		state_input = tf.placeholder("float",[None,state_size])

		W1 = self.variable([state_size,layer1_size],state_size)
		b1 = self.variable([layer1_size],state_size)
		W2 = self.variable([layer1_size,layer2_size],layer1_size)
		b2 = self.variable([layer2_size],layer1_size)
		W3 = self.variable([layer2_size,action_size],0.0003)
		b3 = self.variable([action_size],0.0003)

		layer1 = tf.nn.relu(tf.matmul(state_input,W1) + b1)
		layer2 = tf.nn.relu(tf.matmul(layer1,W2) + b2)
		action_output = tf.matmul(layer2,W3) + b3

		return state_input,W1,b1,W2,b2,W3,b3,action_output

	def train(self,q_gradient_batch,state_batch):
		self.sess.run(self.optimizer,feed_dict={
			self.q_gradient_input:q_gradient_batch,
			self.state_input:state_batch
			})

	def evaluate(self,state_batch):
		return self.sess.run(self.action_output,feed_dict={
			self.state_input:state_batch
			})

	def get_action(self,state):
		return self.sess.run(self.action_output,feed_dict={
			self.state_input:[state]
			})[0]


	def target_evaluate(self,state_batch):
		return self.sess.run(self.target_action_output,feed_dict={
			self.target_state_input:state_batch
			})

	def update_target(self):
		self.sess.run([
			self.target_W1.assign(self.target_W1*(1-TAO)+self.W1*TAO),
			self.target_b1.assign(self.target_b1*(1-TAO)+self.b1*TAO),
			self.target_W2.assign(self.target_W2*(1-TAO)+self.W2*TAO),
			self.target_b2.assign(self.target_b2*(1-TAO)+self.b2*TAO),
			self.target_W3.assign(self.target_W3*(1-TAO)+self.W3*TAO),
			self.target_b3.assign(self.target_b3*(1-TAO)+self.b3*TAO),
			])

	# f fan-in size
	def variable(self,shape,f):
		return tf.Variable(tf.random_uniform(shape,-1/math.sqrt(f),1/math.sqrt(f)))


	def save_network(self,time_step):
		self.saver.save(self.session, 'saved_networks/' + 'actor-network', global_step = time_step)



		
