import numpy as np

from datasets.DataSet import DataSet
from util import get_unique_deterministic


class TinyShakespeareDataSet(DataSet):
    def __init__(self, filepath=None):
        with open(filepath or 'datasets/tiny-shakespeare.txt', 'r') as f:
            self.data = list(f.read())

        print("Data loaded: {} characters".format(len(self.data)))

        self.vocab = get_unique_deterministic(self.data)

    def get_training_samples(self, max_timesteps):
        start_idx = np.random.randint(len(self.data) - max_timesteps)
        return {'x': self.data[start_idx:start_idx + max_timesteps],
                't': self.data[start_idx + 1:start_idx + max_timesteps + 1]}

    def get_primer(self, length):
        start_idx = np.random.randint(len(self.data) - length)
        return self.data[start_idx:start_idx + length]


if __name__ == '__main__':
    dataset = TinyShakespeareDataSet('tiny-shakespeare.txt')
    print(dataset.get_training_samples(10))
    print(dataset.get_primer(length=10))
