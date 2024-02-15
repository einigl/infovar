import itertools as itt
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Union, Optional, Sequence, Callable

import numpy as np

from ..stats.statistics import Statistic, MI, Condh, Corr, LinearInfo
from ..stats.resampling import Resampling, Bootstrapping, Subsampling

__all__ = [
    "Handler"
]


class Handler(ABC):

    ref_path: str
    save_path: str

    variables: Optional[np.ndarray]
    targets: Optional[np.ndarray]
    variable_names: Optional[List[str]]
    target_names: Optional[List[str]]

    stats: Dict[str, Callable]
    resamplings: Dict[str, Resampling]={},

    fn_bounds: Optional[List[Optional[Callable]]]
    inv_fn_bounds: Optional[List[Optional[Callable]]]

    def __init__(
        self
    ):
        self.stats = {
            'mi': MI(),
            'condh': Condh(),
            'corr': Corr(),
            'linearinfo': LinearInfo()
        }

        self.resamplings = {
            'bootstrapping': Bootstrapping(),
            'subsampling': Subsampling(),
        }

        self.ref_path = None
        self.save_path = None

        self.variables = None
        self.targets = None
        self.variable_names = None
        self.target_names = None

        self.additional_stats = None
        self.additional_resamplings = None

        self.fn_bounds = None
        self.inv_fn_bounds = None


    # Setter
        
    def set_paths(
        self,
        ref_path: Optional[str]=None,
        save_path: Optional[str]=None
    ) -> None:
        self.ref_path = ref_path
        self.save_path = save_path
        
    def set_data(
        self,
        variables: np.ndarray,
        targets: np.ndarray,
        variable_names: Union[List[str], str],
        target_names: Union[List[str], str],
    ) -> None:
        assert isinstance(variables, np.ndarray)
        assert isinstance(targets, np.ndarray)

        if isinstance(variable_names, str):
            variable_names = [variable_names]
        if isinstance(target_names, str):
            target_names = [target_names]

        assert variables.ndim in (1, 2)
        assert targets.ndim in (1, 2)
        if variables.ndim == 1:
            variables = np.expand_dims(variables, 1)
        if targets.ndim == 1:
            targets = np.expand_dims(targets, 1)

        assert len(variable_names) == variables.shape[1]
        assert len(target_names) == targets.shape[1]
        assert variables.shape[0] == targets.shape[0]

        self.variables = variables
        self.targets = targets
        self.variable_names = variable_names
        self.target_names = target_names

    def set_additional_stats(
        self,
        additional_stats: Dict[str, Callable]={},
    ) -> None:
        assert all([isinstance(el, Statistic) for el in additional_stats])
        self.stats.update(additional_stats)

    def set_additional_resamplings(
        self,
        additional_resamplings: Dict[str, Callable]={},
    ) -> None:
        assert all([isinstance(el, Resampling) for el in additional_resamplings])
        self.stats.update(additional_resamplings)

    def set_fn(
        self,
        fn_bounds: Optional[List[Optional[Callable]]]=None,
        inv_fn_bounds: Optional[List[Optional[Callable]]]=None
    ) -> None:
        if fn_bounds is None and inv_fn_bounds is None:
            self.fn_bounds = None
            self.inv_fn_bounds = None
            return

        assert self.target_names is not None

        if fn_bounds is None:
            fn_bounds = [None] * len(self.target_names)
        if fn_bounds is None:
            fn_bounds = [None] * len(self.target_names)

        self.fn_bounds = fn_bounds
        self.inv_fn_bounds = inv_fn_bounds


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

    def _filter_data(
        self,
        vars: Union[str, List[str]],
        targs: Union[str, List[str]],
        ranges: Dict[str, Optional[Tuple[Optional[float], Optional[float]]]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess data to be within a given range of data.
        """
        if isinstance(vars, str):
            vars = [vars]
        if isinstance(targs, str):
            targs = [targs]

        vars_idx = [self.variable_names.index(v) for v in vars]
        _vars = np.column_stack([
            self.variables[:, i] for i in vars_idx
        ])

        targs_idx = [self.target_names.index(v) for v in targs]
        _targs = np.column_stack([
            self.targets[:, i] for i in targs_idx
        ])
        
        filt = np.ones(_targs.shape[0], dtype="bool")

        # Remove non-finite pixels in variables
        filt &= np.isfinite(_vars).all(axis=1)

        # Remove pixels out of the targets ranges (including NaNs)
        for rgs in ranges:
            if self.fn_bounds is None:
                fn_b = None
            else:
                fn_b = self.fn_bounds[self.target_names.index(rgs)]

            if fn_b is None:
                fn_b = lambda t: t

            if ranges[rgs] is None:
                pass
            else:
                a, b = ranges[rgs]
                i = self.target_names.index(rgs)
                if a is not None:
                    filt &= fn_b(a) <= self.targets[:, i]
                if b is not None:
                    filt &= self.targets[:, i] <= fn_b(b)

        _vars = _vars[filt]
        _targs = _targs[filt]
       
        return _vars, _targs


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
        print("Variables:")
        for i, v in enumerate(self.variable_names):
            print(f"\t{v}: [{self.variables[:, i].min():.2f}, {self.variables[:, i].max():.2f}]")
        
        print()

        print("Targets:")
        for i, t in enumerate(self.target_names):
            print(f"\t{t}: [{self.targets[:, i].min():.2f}, {self.targets[:, i].max():.2f}]")

        print()

        print("Number of samples:", f"{self.targets.shape[0]:,}")

    @abstractmethod
    def __str__(self):
        pass
