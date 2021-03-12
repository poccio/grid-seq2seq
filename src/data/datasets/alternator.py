import hydra
import numpy as np
import random
from typing import Iterator, List, Dict

from torch.utils.data import IterableDataset


class AlternatorIterableDataset(IterableDataset):

    def __init__(self, datasets: List[Dict], p: List[float], **kwargs):
        assert len(datasets) == len(p)
        self.datasets: List[IterableDataset] = [hydra.utils.instantiate(d, **kwargs) for d in datasets]
        self.p = p
        assert np.isclose(sum(self.p), 1.0)

    def __iter__(self) -> Iterator:

        done = [False for _ in self.datasets]
        iterators = [iter(d) for d in self.datasets]

        while True:

            i = random.choices(list(range(len(self.datasets))), weights=self.p, k=1)[0]

            try:
                batch = next(iterators[i])
            except StopIteration:
                done[i] = True
                if all(done):
                    break
                iterators[i] = iter(self.datasets[i])
                batch = next(iterators[i])

            yield batch
