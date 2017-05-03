import numpy as np

from datasets.DataSet import DataSet
from util import get_unique_deterministic


class TinyShakespeareDataSet(DataSet):
    def __init__(self, filepath=None):
        with open(filepath or 'datasets/tiny-shakespeare.txt', 'r') as f:
            self._data = list(f.read())
        print("Data loaded: {} characters".format(len(self._data)))

        self._vocab = get_unique_deterministic(self._data)
        self._data = np.array(self._data)

    def get_training_samples(self, batch_size, max_timesteps):
        start_idx = np.random.randint(len(self._data) - max_timesteps, size=[batch_size])
        idx = np.tile(np.arange(max_timesteps), reps=[batch_size, 1]) + np.expand_dims(start_idx, axis=1)

        return {'x': self._data[idx],
                't': self._data[idx + 1]}

    def get_primer(self, length):
        start_idx = np.random.randint(len(self._data) - length, size=[1])
        idx = np.expand_dims(np.arange(length) + start_idx, axis=0)

        return self._data[idx]


if __name__ == '__main__':
    dataset = TinyShakespeareDataSet('tiny-shakespeare.txt')
    print(dataset.get_training_samples(batch_size=2, max_timesteps=10))
    print(dataset.get_primer(length=10))
    print(dataset.vocab)
