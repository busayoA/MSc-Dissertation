import tensorflow as tf
class RNN:
    def __init__(self, n_layers):
        self.params = []
        self.W1 = tf.Variable(tf.random.normal([n_layers[0], n_layers[1]], stddev=0.1),name='W1')
		#         self.b1 = tf.Variable(tf.random.normal([n_layers[1]], mean=0.0, stddev=0.1, dtype=tf.dtypes.float32, seed=0), name='b1')
        self.b1 = tf.Variable(tf.zeros([1, n_layers[1]]))
        self.W2 = tf.Variable(tf.random.normal([n_layers[1], n_layers[2]], stddev=0.1),name='W2')
		# self.b2 = tf.Variable(tf.random.normal([n_layers[2]], mean=0.0, stddev=0.1, dtype=tf.dtypes.float32, seed=0), name='b2')
        self.b2 = tf.Variable(tf.zeros([1, n_layers[2]]))
        self.W3 = tf.Variable(
			tf.random.normal([n_layers[2], n_layers[3]],stddev=0.1),
			name='W3')
		#         self.b3 = tf.Variable(tf.random.normal([n_layers[3]], mean=0.0, stddev=0.1, dtype=tf.dtypes.float32, seed=0), name='b3')
        self.b3 = tf.Variable(tf.zeros([1, n_layers[3]]))
		
		# Collect all initialized weights and biases in self.params
        self.params = [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]
    
    def forward(self, x):
        X_tf = tf.cast(x, dtype=tf.float32)
        Z1 = tf.matmul(X_tf, self.W1) + self.b1
        Z1 = tf.nn.relu(Z1)
        Z2 = tf.matmul(Z1, self.W2) + self.b2
        Z2 = tf.nn.relu(Z2)
        Z3 = tf.matmul(Z2, self.W3) + self.b3		
        return Z3