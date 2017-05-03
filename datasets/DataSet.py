from abc import *


class DataSet():
    @abstractmethod
    def __init__(self, filepath=None):
        pass

    @abstractmethod
    def get_training_samples(self, batch_size, max_timesteps):
        pass

    @abstractmethod
    def get_primer(self, length):
        pass

    @property
    def vocab(self):
        return self._vocab
