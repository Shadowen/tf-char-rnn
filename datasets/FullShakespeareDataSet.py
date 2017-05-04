import numpy as np

from datasets.DataSet import DataSet
from util import get_unique_deterministic


class FullShakespeareDataSet(DataSet):
    def __init__(self, filepath=None):
        with open(filepath or 'datasets/shakespeare_input.txt', 'r') as f:
            self._data = list(f.read())
        print("Data loaded: {} characters".format(len(self._data)))

        self._vocab = get_unique_deterministic(self._data)
        self._data = np.array(self._data)

        validation_length = self._data.shape[0] // 10
        self._validation_data = self._data[-validation_length:]
        self._data = self._data[:-validation_length]

        self._epoch = None

    def get_training_samples(self, batch_size, max_timesteps, epoch=None):
        if epoch is None or epoch != self._epoch:
            self.start_idx = np.random.randint(len(self._data) - max_timesteps, size=[batch_size])
            self._epoch = epoch

        idx = np.tile(np.arange(max_timesteps), reps=[batch_size, 1]) + np.expand_dims(self.start_idx, axis=1)
        self.start_idx = (self.start_idx + max_timesteps) % (len(self._data) - max_timesteps)

        return self._data[idx], self._data[idx + 1]

    def get_validation_samples(self, batch_size, max_timesteps):
        idx = np.tile(np.arange(max_timesteps), reps=[batch_size, 1]) + \
              np.expand_dims(np.random.randint(len(self._validation_data) - max_timesteps, size=[batch_size]), axis=1)
        return self._validation_data[idx], self._validation_data[idx + 1]

    def get_primer(self, length):
        start_idx = np.random.randint(len(self._data) - length, size=[1])
        idx = np.expand_dims(np.arange(length) + start_idx, axis=0)

        return self._data[idx]


if __name__ == '__main__':
    dataset = FullShakespeareDataSet('shakespeare_input.txt')
    print(dataset.get_training_samples(batch_size=2, max_timesteps=10))
    print(dataset.get_primer(length=10))
    print(dataset.vocab)
