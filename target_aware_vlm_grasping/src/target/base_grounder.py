from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from utils.data_types import DatasetSample, TargetRegion


class BaseTargetGrounder(ABC):
    @abstractmethod
    def predict(self, sample: DatasetSample, rgb_image: np.ndarray) -> TargetRegion:
        ...

