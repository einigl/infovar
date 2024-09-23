import os
import shutil
import itertools
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Callable, Optional, Tuple

import numpy as np

from ..stats import resampling as rsp
from ..stats import statistics as stt

__all__ = ["Handler"]


class Handler(ABC):
    """
    Abstract class for handlers.
    """

    save_path: str #: Save directory.
    ext: str #: Save files extension.

    getter: Callable[
    [List[str], List[str], Dict[str, Tuple[float, float]]],
    Tuple[np.ndarray, np.ndarray],
    ] #: Function providing samples for further computation.

    stats: Dict[str, Callable] #: Dictionnary of available statistics.
    resamplings: Dict[str, rsp.Resampling] #: Dictionnary of available resamplings.

    def __init__(self):
        self.stats = {
            "mi": stt.MI(),
            "condh": stt.Condh(),
            "corr": stt.Corr(),
            "gaussinfo": stt.GaussInfo(),
            "gaussinforeparam": stt.GaussInfoReparam(),
        }

        self.resamplings = {
            "bootstrapping": rsp.Bootstrapping(),
            "subsampling": rsp.Subsampling(),
        }

        self.save_path = None

        self.getter = None

        self.additional_stats = None
        self.additional_resamplings = None

    # Setter

    def set_path(self, save_path: Optional[str] = None) -> None:
        """
        Defines a new path to a save directory. Must be called at least once before calling other functions such as `store`.

        Parameters
        ----------
        save_path : Optional[str], optional
            New save path. Default None.
        """
        self.save_path = save_path

    def set_getter(
        self,
        getter: Callable[
            [List[str], List[str], Dict[str, Tuple[float, float]]],
            Tuple[np.ndarray, np.ndarray]
        ],
    ) -> None:
        """
        Defines the function (getter) that provides samples for statistical relationship calculations. In most cases, this will correspond to the `get` method of the `StandardGetter`, but users can define their own implementation.

        Parameters
        ----------
        getter : Callable[ [List[str], List[str], Dict[str, Tuple[float, float]]], Tuple[np.ndarray, np.ndarray] ]
            Function providing samples for further computation.
        """
        assert getter is not None
        self.getter = getter

    def set_additional_stats(
        self,
        additional_stats: Dict[str, stt.Statistic] = {},
    ) -> None:
        """
        Add new resamplings (instances of Statistic) to estimate the informativity of variables. Each statistic has a user-defined name. This name will then be reused, for example in the `store` function and its variants.

        Parameters
        ----------
        additional_stats : Dict[str, Statistic], optional
            Additional statistics to be used. Default {}.
        """
        assert all([isinstance(el, stt.Statistic) for el in additional_stats.values()])
        self.stats.update(additional_stats)

    def set_additional_resamplings(
        self,
        additional_resamplings: Dict[str, rsp.Resampling] = {},
    ) -> None:
        """
        Add new resamplings (instances of Resampling) to estimate the variance of some estimators. Each resampling has a user-defined name. This name will then be reused, for example in the `store` function and its variants.

        Parameters
        ----------
        additional_resamplings : Dict[str, Resampling], optional
            Additional resamplings to be used. Default: {}.
        """
        assert all(
            [isinstance(el, rsp.Resampling) for el in additional_resamplings.values()]
        )
        self.stats.update(additional_resamplings)

    # Saves

    @abstractmethod
    def get_filename(self, *args, **kwargs) -> str:
        """
        Builds a save filename from data names.

        Returns
        -------
        str
            Filename.
        """
        pass

    @abstractmethod
    def parse_filename(self, filename: str) -> Any:
        """
        Identifies data names from save filename.

        Parameters
        ----------
        filename : str
            Save filename.

        Returns
        -------
        Any
            Data names.
        """
        pass

    def get_existing_saves(self) -> List[str]:
        """
        Returns the filenames (basenames) of any existing saves at `self.save_path`.
        Any file ending with "cls.ext" is considered a valid save.

        Returns
        -------
        List[str]
            Existing saves.
        """
        if not os.path.exists(self.save_path):
            return []
        return [f for f in os.listdir(self.save_path) if f.endswith(self.ext)]

    # Creation/removal

    def create(self):
        """
        Creates `self.save_path` directory if not exists.
        """
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    @abstractmethod
    def remove(self, *args, **kwargs) -> None:
        """
        Removes `self.save_path` directory if exists.
        """
        if os.path.exists(self.save_path):
            shutil.rmtree(self.save_path)
        else:
            raise FileNotFoundError(f"Save directory {self.save_path} does not exist")

    @abstractmethod
    def delete_stats(self, *args, **kwargs) -> None:
        """
        Removes saved results for a given statistics.
        """
        pass

    # Writing access

    def update(
        self,
        *args,
        **kwargs
    ) -> None:
        """
        Calls `self.store` method with `overwrite=False`.
        """
        self.store(*args, **kwargs, overwrite=False)

    def overwrite(
        self,
        *args,
        **kwargs
    ) -> None:
        """
        Calls `self.store` method with `overwrite=True`.
        """
        self.store(*args, **kwargs, overwrite=True)

    @abstractmethod
    def store(
        self,
        overwrite: bool = False,
        **kwargs
    ) -> None:
        """
        Compute and save values. The behavior depends on the values of the `overwrite` argument.

        Parameters
        ----------
        overwrite : bool, optional
            If True, overwrite the current computed value, if exists. Default: False.
        """
        pass

    # Reading access

    @abstractmethod
    def read(
        self,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Accesses saved values.

        Returns
        -------
        Dict[str, Any]
            Read entries.
        """
        pass

    # Display

    def overview(self):
        """
        Describes the handler and existing backups.
        """
        print(str(self))
        print("Save path:", self.save_path)
        print("Existing saves:")
        for filename in self.get_existing_saves():
            print(filename)

    @abstractmethod
    def __str__(self):
        pass

    # Helpers

    @staticmethod
    def _check_dict_type(
        d: Dict, key_type: Any, value_type: Any
    ) -> Optional[Tuple[Any, Any]]:
        """
        Returns the first entry that does not match the types given as arguments or None if all entries are valid.

        Parameters
        ----------
        d : Dict
            Dictionnary to test.
        key_type : Any
            Expected key type.
        value_type : Any
            Expected value type.

        Returns
        -------
        Optional[Tuple[Any, Any]]
            First entry that does not match the types given as arguments. None if all entries are valid.
        """
        assert isinstance(d, Dict)
        for k, v in d.items():
            if not isinstance(k, key_type) or not isinstance(v, value_type):
                return k, v
        return None

    @staticmethod
    def drop_duplicates(ls: List[List[str]]):
        ls = ls.copy()
        ls.sort()
        return list(k for k,_ in itertools.groupby(ls))
