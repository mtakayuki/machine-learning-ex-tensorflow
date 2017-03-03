import io
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from utils import subdir


flags = tf.app.flags
flags.DEFINE_integer('max_iterations', 1500, 'Max iteration number')
flags.DEFINE_integer('alpha', 0.01, 'Learning rate for gradient descent')
flags.DEFINE_string('logdir', 'logs/ex1/', 'Directory for logs')
FLAGS = flags.FLAGS


class LinearRegression:
    def __init__(self):
        self._read_data()
        self._build_graph()

    def run(self):
        with tf.Session() as self.sess:
            self._initialize()

            # compute and display initial cost
            print(self._loss())

            self._train()

            # print theta to screen
            print('Theta found by gradient descent: ',
                  self.sess.run([self.bias, self.weight]))

            # Plot the linear fit
            self._plot_data_and_linear_fit()

            # Predict values for population sizes of 35,000 and 70,000
            predict1 = self._predict(3.5)
            print('For population = 35,000, we predict a profit of %f' %
                  (predict1 * 10000))
            predict2 = self._predict(7.0)
            print('For population = 70,000, we predict a profit of %f' %
                  (predict2 * 10000))

    def _feed_dict(self):
        return {self.X: self.input_X, self.y: self.input_y}

    def _loss(self):
        return self.sess.run(self.loss, self._feed_dict())

    def _predict(self, x):
        return self.sess.run(self.h, {self.X: x})

    def _read_data(self):
        data = np.genfromtxt('data/ex1/ex1data1.txt', delimiter=',')
        self.input_X = data[:, 0]
        self.input_y = data[:, 1]

    def _build_graph(self):
        self.X = tf.placeholder(tf.float32, name='X')
        self.y = tf.placeholder(tf.float32, name='y')

        self.weight = tf.Variable(0.0, dtype=tf.float32, name='weight')
        self.bias = tf.Variable(0.0, dtype=tf.float32, name='bias')

        with tf.name_scope('hypothesis'):
            self.h = self.weight * self.X + self.bias
        with tf.name_scope('cost'):
            square_delta = tf.square(self.h - self.y)
            self.loss = tf.reduce_mean(square_delta) / 2
        tf.summary.scalar('loss', self.loss)

    def _initialize(self):
        init = tf.global_variables_initializer()
        self.sess.run(init)

        # run gradient descent
        optimizer = tf.train.GradientDescentOptimizer(FLAGS.alpha)
        self.train = optimizer.minimize(self.loss)

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

    def _plot_data_and_linear_fit(self):
        predicts = self._predict(self.input_X)

        plt.figure()
        plt.plot(self.input_X, self.input_y, 'rx', markersize=10,
                 label='Train data')
        plt.xlabel('Population of City in 10,000s')
        plt.ylabel('Profit in $10,000s')

        plt.plot(self.input_X, predicts, '-', label='Linear regression')
        plt.legend()

        self._write_image('plot')

    def _write_image(self, name):
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        image = tf.expand_dims(image, 0)
        summary_op = tf.summary.image(name, image)
        summary = self.sess.run(summary_op)
        self.writer.add_summary(summary, FLAGS.max_iterations)


def main(_):
    LinearRegression().run()


if __name__ == '__main__':
    tf.app.run()
