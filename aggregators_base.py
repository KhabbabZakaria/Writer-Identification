from abc import abstractmethod

import torch


from abc import ABCMeta
from copy import deepcopy

from typing import Dict

class ModuleBase:
    """
    The base class of all classes. You can access default hyper-parameters by Class. And
    set hyper-parameters for each instance at the initialization.
    """
    __metaclass__ = ABCMeta
    default_hyper_params = dict()

    def __init__(self, hps: Dict or None = None):
        """
        Args:
            hps (dict): default hyper parameters in a dict (keys, values).
        """
        # copy hyper_params from class attribute.
        self._hyper_params = deepcopy(self.default_hyper_params)
        if hps is not None:
            self._set_hps(hps)

    def __setattr__(self, key, value) -> None:
        assert key != "hyper_params", "default Hyper-Parameters can not be set in each instance"
        self.__dict__[key] = value

    def get_hps(self) -> Dict:
        return self._hyper_params

    def _set_hps(self, hps: Dict or None = None):
        for key in hps:
            if key not in self._hyper_params:
                raise KeyError
            self._hyper_params[key] = hps[key]


class AggregatorBase(ModuleBase):
    r"""
    The base class for feature aggregators.
    """
    default_hyper_params = dict()

    def __init__(self, hps: Dict or None = None):
        """
        Args:
            hps (dict): default hyper parameters in a dict (keys, values).
        """
        super(AggregatorBase, self).__init__(hps)

    @abstractmethod
    def __call__(self, features: Dict[str, torch.tensor]) -> Dict[str, torch.tensor]:
        pass