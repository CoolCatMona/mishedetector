"""Siamese Neural Network."""
import logging
import os
from enum import IntEnum

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import (Conv2D, Dense, Flatten, Input, Lambda,
                                     MaxPool2D)
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import SGD

from process_images import ImageLoader

logger = logging.getLogger(__name__)


class Channels(IntEnum):
    GREYSCALE = 1
    RGB = 3


class SNN:
    """
    Siamese Neural Network.
    """
    def __init__(self, dataset_path, negatives_path, evaluation_path):
        """Create Siamese Neural Network Architecture

        Args:
            dataset_path ([str]): Path to Positive/Anchor Dataset
            negatives_path ([str]): Path to Negative Dataset
            evaluation_path ([str]): Path to Evaluation Dataset (Images we are evaluating)

        Attributes:
            _DIMEN ([int]): Dimensionality of image, i.e 256 x 256, 128 x 128, etc.
            _COLOR_CHANNELS ([int]): Number of color channels
            input_shape ([int, int, int]): Shape of input, usually _DIMEN x _DIMEN x _COLOR_CHANNELS
            learning_rate ([float]): Learning rate for gradient descent (defaults to .01)
            model ([tensorflow.keras.models.Sequential]): Siamese Neural Network Model
            image_loader ([ImageLoader]): Class that handles loading of images as well as splitting of datasets
        """
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        self._DIMEN = 256
        self._COLOR_CHANNELS = Channels.RGB
        self.input_shape = (self._DIMEN, self._DIMEN, self._COLOR_CHANNELS)
        self.learning_rate = .01
        self._model = None
        self._make_snn()
        self.image_loader = ImageLoader(dataset_path=dataset_path,
                                        negatives_path=negatives_path,
                                        evaluation_path=evaluation_path)

    def _make_snn(self):
        """
        Constructs Architecture and stores in class
        """
        self.conv_net = Sequential()
        # NOTE: First Convolutional layer usually has a larger kernel size
        self.conv_net.add(Conv2D(filters=64, kernel_size=(10, 10),
                                 activation='relu', input_shape=self.input_shape))
        self.conv_net.add(MaxPool2D())

        self.conv_net.add(Conv2D(filters=128, kernel_size=(7, 7),
                                 activation='relu'))
        self.conv_net.add(MaxPool2D())

        self.conv_net.add(Conv2D(filters=128, kernel_size=(4, 4),
                                 activation='relu'))
        self.conv_net.add(MaxPool2D())

        self.conv_net.add(Conv2D(filters=256, kernel_size=(4, 4),
                                 activation='relu'))

        self.conv_net.add(Flatten())
        self.conv_net.add(Dense(units=4096, activation='sigmoid'))

        input_image_1 = Input(self.input_shape)
        input_image_2 = Input(self.input_shape)

        # The CNN Defined earlier will return embeddings that we use to compute distance
        encoded_image_1 = self.conv_net(input_image_1)
        encoded_image_2 = self.conv_net(input_image_2)

        euclidian_distance_layer = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
        euclidian_distance = euclidian_distance_layer([encoded_image_1, encoded_image_2])

        output = Dense(units=1, activation='sigmoid')(euclidian_distance)
        self.model = Model(inputs=[input_image_1, input_image_2], outputs=output)

        optimizer = SGD(learning_rate=self.learning_rate)
        self.model.compile(loss='binary_crossentropy', metrics=['binary_accuracy'], optimizer=optimizer)

    def train_snn(self, number_of_iterations, final_momentum, momentum_slope, evaluate_each, model_name):
        """Train Siamese Neural Network
        In each every evaluate_each train iterations we evaluate one-shot tasks in
        validation and evaluation set.
        Arguments:
            number_of_iterations: maximum number of iterations to train.
            final_momentum: mu_j in the paper. Each layer starts at 0.5 momentum
                but evolves linearly to mu_j
            momentum_slope: slope of the momentum evolution. In the paper we are
                only told that this momentum evolves linearly.
            evaluate each: number of iterations defined to evaluate the one-shot
                tasks.
            model_name: save_name of the model
        Returns:
            Evaluation Accuracy
        """
        self.image_loader.split_train_datasets()

        # Variables that will store 100 iterations losses and accuracies
        # after evaluate_each iterations these will be passed to tensorboard logs
        train_losses = np.zeros(shape=(evaluate_each))
        train_accuracies = np.zeros(shape=(evaluate_each))
        count = 0
        # Stop criteria variables
        best_validation_accuracy = 0.0
        best_accuracy_iteration = 0
        validation_accuracy = 0.0

        # Train loop
        for iteration in range(number_of_iterations):

            # train set
            images, labels = self.image_loader.get_train_batch()
            train_loss, train_accuracy = self.model.train_on_batch(
                images, labels)

            # Decay learning rate 1 % per 500 iterations (in the paper the decay is
            # 1% per epoch). Also update linearly the momentum (starting from 0.5 to 1)
            if (iteration + 1) % 500 == 0:
                K.set_value(self.model.optimizer.lr, K.get_value(
                    self.model.optimizer.lr) * 0.99)
            if K.get_value(self.model.optimizer.momentum) < final_momentum:
                K.set_value(self.model.optimizer.momentum, K.get_value(
                    self.model.optimizer.momentum) + momentum_slope)

            train_losses[count] = train_loss
            train_accuracies[count] = train_accuracy

            # validation set
            count += 1
            logger.info(f"Iteration {iteration + 1}/{number_of_iterations}:\
                          Train loss: {train_loss}, Train Accuracty: {train_accuracy},\
                          Learning Rate: {K.get_value(self.model.optimizer.lr)}")

            # Each 100 iterations perform a one_shot_task and write to tensorboard the
            # stored losses and accuracies
            if (iteration + 1) % evaluate_each == 0:
                number_of_runs = 1  # MAGIC NUMBER
                validation_accuracy = self.image_loader.one_shot_test(
                    self.model, number_of_runs, is_validation=True)

                count = 0

                # Some hyperparameters lead to 100%, although the output is almost the same in
                # all images.
                if (validation_accuracy == 1.0 and train_accuracy == 0.5):
                    logger.info('Early Stopping: Gradient Explosion')
                    logger.info(f'Validation Accuracy = {best_validation_accuracy}')
                    return 0
                elif train_accuracy == 0.0:
                    return 0
                else:
                    # Save the model
                    if validation_accuracy > best_validation_accuracy:
                        logger.info(f"Validation accuracy increased! Was {best_validation_accuracy}\
                             is now {validation_accuracy}! An increase of\
                                  {validation_accuracy - best_validation_accuracy}")
                        best_validation_accuracy = validation_accuracy
                        best_accuracy_iteration = iteration

                        model_json = self.model.to_json()

                        if not os.path.exists('./models'):
                            os.makedirs('./models')
                        with open('models/' + model_name + '.json', "w") as json_file:
                            json_file.write(model_json)
                        self.model.save_weights('models/' + model_name + '.h5')

            # If accuracy does not improve for 10000 batches stop the training
            if iteration - best_accuracy_iteration > 1000:
                logger.info('Early Stopping: validation accuracy did not increase for 1000 iterations')
                logger.info(f'Best Validation Accuracy = {best_validation_accuracy}')
                break

        logger.info('Trained Ended!')
        return best_validation_accuracy
