"""Processes images in dataset."""
from enum import IntEnum
import logging
import os
import random

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class Channels(IntEnum):
    GREYSCALE = 1
    RGB = 3


class ImageLoader():
    def __init__(self, dataset_path, negatives_path, evaluation_path):
        """Class for loading and preparing of data.

        Args:
            dataset_path ([str]): Path to Positive/Anchor Dataset
            negatives_path ([str]): Path to Negative Dataset
            evaluation_path ([str]): Path to Evaluation Dataset (Images we are evaluating)
        Attributes:
            dataset_path ([str]): Path to Positive/Anchor Dataset
            images ([list]): List of images in dataset
            _num_images ([int]): Number of images in dataset
            negatives_path ([str]): Path to Negative Dataset
            negative_images ([list]): List of images in dataset
            _num_negatives ([int]): Number of images in dataset
            evaluation_path ([str]): Path to Evaluation Images
            evaluation_images ([list]): List of images in dataset
            _num_evals ([int]): Number of images in dataset
            _train_images ([list]): List of images to train on
            _validation_images ([list]): List of images for validation
            _DIMEN ([int]): Dimensionality of image
            _COLOR_CHANNELS ([Channels]): Number of color channels
        """
        self.dataset_path = dataset_path
        self.images = os.listdir(self.dataset_path)
        self._num_images = len(self.images)

        self.negatives_path = negatives_path
        self.negative_images = os.listdir(self.negatives_path)
        self._num_negatives = len(self.negative_images)

        self.evaluation_path = evaluation_path
        self.evaluation_images = os.listdir(self.evaluation_path)
        self._num_evals = len(self.evaluation_images)

        self._train_images = []
        self._validation_images = []

        self._DIMEN = 256
        self._COLOR_CHANNELS = Channels.RGB

    def _convert_image(self, images, pair, pair_pos):
        """Converts image into numpy array.

        Args:
            images ([list]): List of images
            pair (int): Index of pair of images
            pair_pos (int): Pair position (either 0 or 1)

        Returns:
            [np.array]: Image as array
        """
        image = Image.open(images[pair * 2 + pair_pos])
        image = np.asarray(image).astype(np.float64)
        image = image / image.std() - image.mean()
        return image

    def _get_images_and_labels(self, path_list, is_one_shot=False):
        """Loads images and correspondent labels from input path.
           Reads images and returns pairss of images and labels, for training and validation
           labels are alternating 0's and 1's, for evaluation only first pair has 1.

        Args:
            path_list: List containing paths of images
            is_one_shot (bool, optional): Is one-shot task, or for training. Defaults to False.

        Returns:
            list, list: pairs of images for batch, correspondent lables (-1 for same class, 0 for different class)
        """
        # Iterate through dataset, create image/label pair
        number_of_pairs = int(len(path_list) // 2)
        pairs_of_images = [np.zeros((number_of_pairs,
                                     self._DIMEN,
                                     self._DIMEN,
                                     self._COLOR_CHANNELS)) for i in range(2)]
        labels = np.zeros((number_of_pairs, 1))

        for pair in range(number_of_pairs):
            image = self._convert_image(path_list, pair, 0)
            if self._COLOR_CHANNELS == Channels.GREYSCALE:
                pairs_of_images[0][pair, :, :, 0] = image
            else:
                pairs_of_images[0][pair, :, :] = image

            image = self._convert_image(path_list, pair, 1)
            if self._COLOR_CHANNELS == Channels.GREYSCALE:
                pairs_of_images[1][pair, :, :, 0] = image
            else:
                pairs_of_images[1][pair, :, :] = image
            if not is_one_shot:
                if (pair + 1) % 2 == 0:
                    labels[pair] = 0
                else:
                    labels[pair] = 1

            else:
                if pair == 0:
                    labels[pair] = 1
                else:
                    labels[pair] = 0

        if not is_one_shot:
            random_permutation = np.random.permutation(number_of_pairs)
            labels = labels[random_permutation]
            if self._COLOR_CHANNELS == Channels.GREYSCALE:
                pairs_of_images[0][:, :, :, :] = pairs_of_images[0][random_permutation, :, :, :]
                pairs_of_images[1][:, :, :, :] = pairs_of_images[1][random_permutation, :, :, :]
            else:
                pairs_of_images[0][:, :, :] = pairs_of_images[0][random_permutation, :, :]
                pairs_of_images[1][:, :, :] = pairs_of_images[1][random_permutation, :, :]

        return pairs_of_images, labels

    def split_train_datasets(self):
        """ Splits the train set in train and validation
        Divide the n images in train and validation with an 80/20 split.
        """
        available_images = self.images

        train_indexes = random.sample(
            range(0, self._num_images - 1), int(0.8 * self._num_images))

        # Sort the indexes in reverse order for easy popping while not changing index
        train_indexes.sort(reverse=True)

        for index in train_indexes:
            self._train_images.append(available_images[index])
            available_images.pop(index)

        # The remaining images are saved for validation
        self._validation_images = available_images

    def get_train_batch(self):
        """Loads and returns a batch of train images

        Returns:
            pairs_of_images: pairs of images for the current batch
            labels: correspondent labels -1 for same class, 0 for different classes
        """
        bacth_images_path = []

        # Get Anchor and Positive Image
        image_indexes = random.sample(range(0, len(self._train_images)), 3)
        image = os.path.join(
            self.dataset_path, self._train_images[image_indexes[0]])
        bacth_images_path.append(image)
        image = os.path.join(
            self.dataset_path, self._train_images[image_indexes[1]])
        bacth_images_path.append(image)

        # Get Anchor and Negative Image
        image = os.path.join(
            self.dataset_path, self._train_images[image_indexes[2]])
        bacth_images_path.append(image)
        image_indexes = random.sample(range(0, self._num_negatives), 1)

        image = os.path.join(
            self.negatives_path, self.negative_images[image_indexes[0]])
        bacth_images_path.append(image)

        images, labels = self._get_images_and_labels(
            bacth_images_path, is_one_shot=False)

        return images, labels

    def get_one_shot_batch(self, is_validation):
        """ Loads and returns a batch for one-shot task images
        Gets a one-shot batch for evaluation or validation set, it consists in a
        single image that will be compared with a support set of images. It returns
        the pair of images to be compared by the model and it's labels (the first
        pair is always 1) and the remaining ones are 0's
        Returns:
            pairs_of_images: pairs of images for the current batch
            labels: correspondent labels -1 for same class, 0 for different classes
        """

        # Set some variables that will be different for validation and evaluation sets
        if is_validation:
            image_folder_path = self.dataset_path
            num_images = len(self._validation_images)
            images = self._validation_images
        else:
            image_folder_path = self.evaluation_path
            num_images = self._num_evals
            images = self.evaluation_images

        bacth_images_path = []

        image_indexes = random.sample(range(0, num_images), 2)

        test_image = os.path.join(
            image_folder_path, images[image_indexes[0]])
        bacth_images_path.append(test_image)
        image = os.path.join(
            image_folder_path, images[image_indexes[1]])
        bacth_images_path.append(image)

        if not is_validation:
            logger.info(f"Using {test_image} as Anchor")
            logger.info(f"Using {image} as Positive")

        for negative in self.negative_images:
            image = os.path.join(self.negatives_path, negative)
            bacth_images_path.append(test_image)
            bacth_images_path.append(image)

        images, labels = self._get_images_and_labels(
            bacth_images_path, is_one_shot=True)

        return images, labels

    def one_shot_test(self, model, number_of_tasks,
                      is_validation):
        """Prepare one-shot task and evaluate its performance
        Make one shot task in validation and evaluation sets
        Returns:
            mean_accuracy: mean accuracy for the one-shot task
        """

        # Set some variables that depend on dataset
        if is_validation:
            logger.info('\nMaking One Shot Task on validation images:')
        else:
            logger.info('\nMaking One Shot Task on evaluation images:')

        mean_accuracy = 0

        for _ in range(number_of_tasks):
            images, _ = self.get_one_shot_batch(is_validation=is_validation)
            probabilities = model.predict_on_batch(images)
            # logger.info(f"Probabilities are {probabilities}")

            # Case where most are same class
            if np.argmax(probabilities) == 0 and probabilities.std() > 0.01:
                accuracy = 1.0
            else:
                # accuracy = 0.0
                accuracy = np.mean(probabilities)
                pass

            mean_accuracy += accuracy

        mean_accuracy /= number_of_tasks

        logger.info(f"Probability that image given is of the class: {mean_accuracy}")

        return mean_accuracy
