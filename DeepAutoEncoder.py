import tensorflow as tf
import os

MODEL_DIR = 'model/'
MODEL_NAME = 'ae'


class DeepAutoEncoder:
    def __init__(self, input_dimension):
        # Training Parameters
        tf.reset_default_graph()
        self.learning_rate = 0.01

        # Network Parameters
        # self.num_hidden_1 = 256  # 1st layer num features
        # self.num_hidden_2 = 128  # 2nd layer num features (the latent dim)
        self.num_hidden_1 = int(1.8 * input_dimension)    # 1st layer num features
        self.num_hidden_2 = int(0.8 * self.num_hidden_1)  # 2nd layer num features
        self.num_hidden_3 = int(0.6 * self.num_hidden_2)  # 3rd layer num features
        self.num_hidden_4 = int(0.5 * self.num_hidden_3)  # 3rd layer num features

        self.dimension = input_dimension
        self.weights = {
            'encoder_h1': tf.Variable(tf.random_normal([self.dimension, self.num_hidden_1])),
            'encoder_h2': tf.Variable(tf.random_normal([self.num_hidden_1, self.num_hidden_2])),
            'encoder_h3': tf.Variable(tf.random_normal([self.num_hidden_2, self.num_hidden_3])),
            'encoder_h4': tf.Variable(tf.random_normal([self.num_hidden_3, self.num_hidden_4])),
            'decoder_h1': tf.Variable(tf.random_normal([self.num_hidden_4, self.num_hidden_3])),
            'decoder_h2': tf.Variable(tf.random_normal([self.num_hidden_3, self.num_hidden_2])),
            'decoder_h3': tf.Variable(tf.random_normal([self.num_hidden_2, self.num_hidden_1])),
            'decoder_h4': tf.Variable(tf.random_normal([self.num_hidden_1, self.dimension])),
        }
        self.biases = {
            'encoder_b1': tf.Variable(tf.random_normal([self.num_hidden_1])),
            'encoder_b2': tf.Variable(tf.random_normal([self.num_hidden_2])),
            'encoder_b3': tf.Variable(tf.random_normal([self.num_hidden_3])),
            'encoder_b4': tf.Variable(tf.random_normal([self.num_hidden_4])),
            'decoder_b1': tf.Variable(tf.random_normal([self.num_hidden_3])),
            'decoder_b2': tf.Variable(tf.random_normal([self.num_hidden_2])),
            'decoder_b3': tf.Variable(tf.random_normal([self.num_hidden_1])),
            'decoder_b4': tf.Variable(tf.random_normal([self.dimension])),
        }

        self.input_X = tf.placeholder("float", name='X', shape=[None, self.dimension])
        encoder_op = self.encoder(self.input_X)
        decoder_op = self.decoder(encoder_op)
        self.output_X = decoder_op

        # Define loss and optimizer, minimize the squared error
        self.loss = tf.reduce_mean(tf.pow(self.input_X - self.output_X, 2), name="loss")
        self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss, name="opt")
        # self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, name="opt")

        # Initialize tensorflow session
        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def encoder(self, x):
        # Encoder Hidden layer with sigmoid activation #1
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, self.weights['encoder_h1']), self.biases['encoder_b1']))
        layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, self.weights['encoder_h2']), self.biases['encoder_b2']))
        layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, self.weights['encoder_h3']), self.biases['encoder_b3']))
        layer_4 = tf.nn.sigmoid(tf.add(tf.matmul(layer_3, self.weights['encoder_h4']), self.biases['encoder_b4']))
        return layer_4

    # Building the decoder
    def decoder(self, x):
        # Decoder Hidden layer with sigmoid activation #1
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, self.weights['decoder_h1']), self.biases['decoder_b1']))
        layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, self.weights['decoder_h2']), self.biases['decoder_b2']))
        layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, self.weights['decoder_h3']), self.biases['decoder_b3']))
        layer_4 = tf.nn.sigmoid(tf.add(tf.matmul(layer_3, self.weights['decoder_h4']), self.biases['decoder_b4']))
        return layer_4

    def train_model(self, batch_X):
        # _, loss = self.sess.run([self.optimizer, self.loss], feed_dict={self.input_X: batch_X})
        _, loss = self.sess.run([self.optimizer, self.loss], feed_dict={self.input_X: batch_X})
        return loss

    def test_model(self, test_X):
        loss = self.sess.run(self.loss, feed_dict={self.input_X: test_X})
        return loss

    def save_model(self):
        # Save model routinely
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)
        # Save the latest trained models
        self.saver.save(self.sess, MODEL_DIR + MODEL_NAME)

    def load_model(self):
        # Restore the trained model
        assert os.path.exists(MODEL_DIR + MODEL_NAME)
        self.saver.restore(self.sess, MODEL_DIR + MODEL_NAME)

        # sess = tf.Session()
        # self.saver = tf.train.import_meta_graph(MODEL_NAME + '.meta')
        # self.saver.restore(sess, tf.train.latest_checkpoint('./'))
        # graph = tf.get_default_graph()
        # input_X = graph.get_tensor_by_name("X:0")
        # loss = graph.get_tensor_by_name("loss:0")
        # opt = graph.get_tensor_by_name("opt:0")
        # _, loss = sess.run([opt, loss], feed_dict={input_X: batch_X})

