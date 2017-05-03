from abc import *


class DataSet():
    @abstractmethod
    def __init__(self, filepath=None):
        pass

    @abstractmethod
    def get_training_samples(self, max_timesteps):
        pass
