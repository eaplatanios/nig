import abc
import os

__author__ = 'Emmanouil Antonios Platanios'


class Learner(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, learning_rate=1e-2, maximum_number_of_iterations=100000,
                 working_dir=os.getcwd()):
        self.learning_rate = learning_rate
        self.maximum_number_of_iterations = maximum_number_of_iterations
        self.working_directory = working_dir

    @abc.abstractmethod
    def train(self, models, train_data, validation_data=None, test_data=None):
        pass

    @abc.abstractmethod
    def predict(self, input_data):
        pass

    @abc.abstractmethod
    def predict_iterator(self, input_data):
        pass


class TensorFlowLearner(Learner):
    """Used for training a single TensorFlow model."""
    def train(self, model, train_data, validation_data=None, test_data=None):
        pass

    def predict(self, input_data):
        pass

    def predict_iterator(self, input_data):
        pass


class TensorFlowMultiModelValidationSetLearner(Learner):
    """Used for training multiple TensorFlow models that have the same input and
    predict the same quantities, using a validation data set to pick the best
    model."""
    def train(self, models, train_data, validation_data=None, test_data=None):
        pass

    def predict(self, input_data):
        pass

    def predict_iterator(self, input_data):
        pass


class TensorFlowMultiModelCrossValidationLearner(Learner):
    """Used for training multiple TensorFlow models that have the same input and
    predict the same quantities, using cross-validation to pick the best
    model."""
    def train(self, models, train_data, validation_data=None, test_data=None):
        pass

    def predict(self, input_data):
        pass

    def predict_iterator(self, input_data):
        pass


class TensorFlowMultiModelNIGLearner(Learner):
    """Used for training multiple TensorFlow models that have the same input and
    predict the same quantities, using the NIG agreement-driven approach to
    train them jointly and allow them to make predictions jointly."""
    def train(self, models, train_data, validation_data=None, test_data=None):
        pass

    def predict(self, input_data):
        pass

    def predict_iterator(self, input_data):
        pass
