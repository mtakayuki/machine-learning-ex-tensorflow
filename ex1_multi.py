import numpy as np
import tensorflow as tf

from utils import subdir


flags = tf.app.flags
flags.DEFINE_integer('max_iterations', 400, 'Max iteration number')
flags.DEFINE_float('alpha', 0.1, 'Learning rate for gradient descent')
flags.DEFINE_string('logdir', 'logs/ex1_multi/', 'Directory for logs')
FLAGS = flags.FLAGS


class LinearRegressionWithMultipleVariables:
    def __init__(self):
        self._read_data()
        self._build_graph()

    def run(self):
        with tf.Session() as self.sess:
            self._initialize()
            self._train()

            # Estimate the price of a 1650 sq-ft, 3 br house
            price = self._predict([1650, 3])
            print('Predicted price of a 1650 sq-ft, 3 br house ',
                  '(using gradient descent):\n $%f\n' % price)

    def _feed_dict(self):
        return {self.X: self.input_X, self.y: self.input_y}

    def _predict(self, x):
        if not hasattr(x, 'reshape'):
            x = np.array(x)
        x = x.reshape(-1, 2)
        return self.sess.run(self.h, {self.X: x})

    def _read_data(self):
        data = np.genfromtxt('data/ex1/ex1data2.txt', delimiter=',')
        self.input_X = data[:, 0:2].astype(np.float32)
        self.input_y = data[:, 2:3].astype(np.float32)

    def _normalize_features(self):
        self.train_X = tf.placeholder(dtype=self.input_X.dtype,
                                      shape=self.input_X.shape, name='train_X')
        self.X = tf.placeholder(tf.float32, shape=[None, 2], name='X')
        self.y = tf.placeholder(tf.float32, shape=[None, 1], name='y')

        with tf.name_scope('normalize_features'):
            mean, var = tf.nn.moments(self.train_X, axes=[0])
            self.mu = tf.Variable(mean, trainable=False,
                                  collections=[], name='mu')
            self.sigma = tf.Variable(tf.sqrt(var), trainable=False,
                                     collections=[], name='sigma')
            self.norm_X = (self.X - self.mu) / self.sigma

    def _build_graph(self):
        self._normalize_features()

        self.weights = tf.Variable(tf.zeros([2, 1]), dtype=tf.float32,
                                   name='weights')
        self.bias = tf.Variable(tf.zeros([1]), dtype=tf.float32, name='bias')

        with tf.name_scope('hypothesis'):
            self.h = tf.matmul(self.norm_X, self.weights) + self.bias

        with tf.name_scope('cost'):
            square_delta = tf.square(self.h - self.y)
            self.loss = tf.reduce_mean(square_delta) / 2

        optimizer = tf.train.GradientDescentOptimizer(FLAGS.alpha)
        self.train = optimizer.minimize(self.loss)

    def _initialize(self):
        self.sess.run([self.mu.initializer, self.sigma.initializer],
                      feed_dict={self.train_X: self.input_X})
        init = tf.global_variables_initializer()
        self.sess.run(init)

        tf.summary.scalar('loss', self.loss)

        self.merged = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(subdir(FLAGS.logdir),
                                            self.sess.graph)

    def _train(self):
        for i in range(FLAGS.max_iterations):
            if i % 10 == 0:
                summary, _ = self.sess.run([self.merged, self.loss],
                                           self._feed_dict())
                self.writer.add_summary(summary, i)
            self.sess.run(self.train, self._feed_dict())


def main(_):
    LinearRegressionWithMultipleVariables().run()


if __name__ == '__main__':
    tf.app.run()
