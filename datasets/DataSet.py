from abc import *


class DataSet():
    @abstractmethod
    def __init__(self, filepath=None):
        pass

    @abstractmethod
    def get_training_samples(self, batch_size, max_timesteps):
        pass

    def get_validation_samples(self, batch_size, max_timesteps):
        raise NotImplementedError('This dataset has no validation samples.')

    @abstractmethod
    def get_primer(self, length):
        pass

    @property
    def vocab(self):
        return self._vocab
