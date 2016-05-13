
import tensorflow as tf
import numpy as np
import math

LAYER1_SIZE = 200
LAYER2_SIZE = 100
LEARNING_RATE = 0.0001
TAO = 0.001

class CriticNetwork:
	"""docstring for CriticNetwork"""
	def __init__(self,state_size = 4,action_size = 2):
		self.graph = tf.Graph()
		with self.graph.as_default():
			self.sess = tf.InteractiveSession()

			# create q network
			self.state_input,\
			self.action_input,\
			self.W1,\
			self.b1,\
			self.W2,\
			self.W2_action,\
			self.b2,\
			self.W3,\
			self.b3,\
			self.q_value_output = self.create_q_network(state_size,action_size)

			# create target q network (the same structure with q network)
			self.target_state_input,\
			self.target_action_input,\
			self.target_W1,\
			self.target_b1,\
			self.target_W2,\
			self.target_W2_action,\
			self.target_b2,\
			self.target_W3,\
			self.target_b3,\
			self.target_q_value_output = self.create_q_network(state_size,action_size)

			# Define training optimizer
			self.y_input = tf.placeholder("float",[None,1])
			self.cost = tf.pow(self.q_value_output-self.y_input,2)/tf.to_float(tf.shape(self.y_input)[0])
			+ 0.0001*tf.reduce_sum(tf.pow(self.W2,2))+0.0001*tf.reduce_sum(tf.pow(self.b2,2))
			self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.cost)

			self.action_gradients = tf.gradients(self.q_value_output,self.action_input)
			#self.action_gradients = [self.action_gradients_v[0]/tf.to_float(tf.shape(self.action_gradients_v[0])[0])]

			self.sess.run(tf.initialize_all_variables())
			# copy target parameters
			self.sess.run([
				self.target_W1.assign(self.W1),
				self.target_b1.assign(self.b1),
				self.target_W2.assign(self.W2),
				self.target_W2_action.assign(self.W2_action),
				self.target_b2.assign(self.b2),
				self.target_W3.assign(self.W3),
				self.target_b3.assign(self.b3)
			])



	def create_q_network(self,state_size,action_size):
		# the layer size could be changed
		layer1_size = LAYER1_SIZE
		layer2_size = LAYER2_SIZE

		state_input = tf.placeholder("float",[None,state_size])
		action_input = tf.placeholder("float",[None,action_size])

		W1 = self.variable([state_size,layer1_size],state_size)
		b1 = self.variable([layer1_size],state_size)
		W2 = self.variable([layer1_size,layer2_size],layer1_size+action_size)
		W2_action = self.variable([action_size,layer2_size],layer1_size+action_size)
		b2 = self.variable([layer2_size],layer1_size+action_size)
		W3 = self.variable([layer2_size,1],0.0003)
		b3 = self.variable([1],0.0003)

		layer1 = tf.nn.relu(tf.matmul(state_input,W1) + b1)
		layer2 = tf.nn.relu(tf.matmul(layer1,W2) + tf.matmul(action_input,W2_action) + b2)
		q_value_output = tf.matmul(layer2,W3) + b3

		return state_input,action_input,W1,b1,W2,W2_action,b2,W3,b3,q_value_output

	def train(self,y_batch,state_batch,action_batch):
		#action = np.transpose([action_batch])

		self.sess.run(self.optimizer,feed_dict={
			self.y_input:np.transpose([y_batch]),
			self.state_input:state_batch,
			self.action_input:action_batch
			})


	def gradients(self,state_batch,action_batch):
		return self.sess.run(self.action_gradients,feed_dict={
			self.state_input:state_batch,
			self.action_input:action_batch
			})[0]

	def target_evaluate(self,state_batch,action_batch):
		return self.sess.run(self.target_q_value_output,feed_dict={
			self.target_state_input:state_batch,
			self.target_action_input:action_batch
			})

	def evaluate(self,state_batch,action_batch):
		return self.sess.run(self.q_value_output,feed_dict={
			self.state_input:state_batch,
			self.action_input:action_batch})

	def update_target(self):
		self.sess.run([
			self.target_W1.assign(self.target_W1*(1-TAO)+self.W1*TAO),
			self.target_b1.assign(self.target_b1*(1-TAO)+self.b1*TAO),
			self.target_W2.assign(self.target_W2*(1-TAO)+self.W2*TAO),
			self.target_W2_action.assign(self.target_W2_action*(1-TAO)+self.W2_action*TAO),
			self.target_b2.assign(self.target_b2*(1-TAO)+self.b2*TAO),
			self.target_W3.assign(self.target_W3*(1-TAO)+self.W3*TAO),
			self.target_b3.assign(self.target_b3*(1-TAO)+self.b3*TAO),
			])

	# f fan-in size
	def variable(self,shape,f):
		return tf.Variable(tf.random_uniform(shape,-1/math.sqrt(f),1/math.sqrt(f)))


	def save_network(self,time_step):
		self.saver.save(self.session, 'saved_networks/' + 'critic-network', global_step = time_step)


