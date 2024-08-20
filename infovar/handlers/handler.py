import itertools as itt
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Union, Optional, Sequence, Callable

import numpy as np

from ..stats import statistics as stt
from ..stats import resampling as rsp

__all__ = [
    "Handler"
]


class Handler(ABC):

    save_path: str

    getter: Callable[[List[str], List[str], Dict[str, Tuple[float, float]]], Tuple[np.ndarray, np.ndarray]]

    stats: Dict[str, Callable]
    resamplings: Dict[str, rsp.Resampling]={}

    def __init__(
        self
    ):
        self.stats = {
            'mi': stt.MI(),
            'condh': stt.Condh(),
            'corr': stt.Corr(),
            'linearinfo': stt.LinearInfo(),
            'linearinforeparam': stt.LinearInfoReparam(),
        }

        self.resamplings = {
            'bootstrapping': rsp.Bootstrapping(),
            'subsampling': rsp.Subsampling(),
        }

        self.save_path = None

        self.getter = None

        self.additional_stats = None
        self.additional_resamplings = None


    # Setter
        
    def set_path(
        self,
        save_path: Optional[str]=None
    ) -> None:
        self.save_path = save_path
        
    def set_getter(
        self,
        getter: Callable[[List[str], List[str], Dict[str, Tuple[float, float]]], Tuple[np.ndarray, np.ndarray]],
    ) -> None:
        assert getter is not None
        self.getter = getter

    def set_additional_stats(
        self,
        additional_stats: Dict[str, stt.Statistic]={},
    ) -> None:
        assert all([isinstance(el, stt.Statistic) for el in additional_stats.values()])
        self.stats.update(additional_stats)

    def set_additional_resamplings(
        self,
        additional_resamplings: Dict[str, Callable]={},
    ) -> None:
        assert all([isinstance(el, rsp.Resampling) for el in additional_resamplings.values()])
        self.stats.update(additional_resamplings)


    # General
        
    @abstractmethod
    def get_filename(
        self,
        targets: Union[str, Sequence[str]],
        variables: Union[str, Sequence[str]]
    ) -> str:
        pass

    @staticmethod
    def _comb(
        ls: Sequence,
        repeats: Union[int, Sequence[int]]
    ) -> List[Sequence]:
        res = []
        for r in repeats:
            res += list(itt.combinations(ls, r))
        return res


    # Writing access

    @abstractmethod
    def create(
        self,
        targets: Union[str, Sequence[str]],
        variables: Union[str, Sequence[str]]
    ):
        pass

    @abstractmethod
    def remove(
        self,
        targets: Optional[Sequence[Union[str, Sequence[str]]]],
        variables: Optional[Sequence[Union[str, Sequence[str]]]]
    ) -> None:
        pass

    @abstractmethod
    def delete_stats(
        self,
        targs: Union[str, List[str]],
        stats: Union[str, List[str]]
    ) -> None:
        pass

    def update(
        self,
        inputs_dict: Dict[str, Any],
    ) -> None:
        """
        We keep the previous results and only recompute the stats that has never been computed before.
        Note that there is no guarantee that the data used.
        An error is raised if the number of samples is different from that already stored, if any.
        """
        self.store(
            inputs_dict,
            overwrite=False
        )

    def overwrite(
        self,
        inputs_dict: Dict[str, Any],
    ) -> None:
        """
        We recompute the stats that has already been calculated.
        Stats already calculated but not included in the inputs_dict are kept.
        An error is raised if the number of samples is different from that already stored, if any.
        """
        self.store(
            inputs_dict,
            overwrite=True
        )

    @abstractmethod
    def store(
        self,
        inputs_dict: Dict[str, Any],
        overwrite: bool=False
    ):
        pass


    # Reading access

    @abstractmethod
    def read(
        self,
        targs: Union[str, Sequence[str]],
        vars: Union[str, Sequence[str]],
        wins_targs: Union[str, Sequence[str]]
    ) -> Dict[str, Any]:
        pass
    

    @abstractmethod
    def _check_inputs(
        self,
        inputs_dict: Dict[str, Any],
        ref_dict: Dict[str, Any]
    ) -> None:
        # TODO
        # raise ValueError("Inputs must match the reference file")
        pass


    # Display

    def overview(self):
        print("Handler")
        print("Save path:", self.save_path)

    @abstractmethod
    def __str__(self):
        pass
