import os
import shutil
import json
import yaml
import itertools as itt
from typing import List, Dict, Any, Tuple, Union, Optional, Sequence, Callable

import numpy as np
from tqdm import tqdm

from ..stats.statistics import Statistic, MI, Condh, Corr


__all__ = [
    "ContinuousHandler"
]

class ContinuousHandler:

    variables: np.ndarray
    targets: np.ndarray
    variable_names: List[str]
    target_names: List[str]
    ref_path: str
    save_path: str
    additional_stats: Dict[str, Callable]
    fn_bounds: List[Optional[Callable]]

    def __init__(
        self,
        variables: np.ndarray,
        targets: np.ndarray,
        variable_names: Union[List[str], str],
        target_names: Union[List[str], str],
        ref_path: str,
        save_path: str,
        additional_stats: Dict[str, Callable]={},
        fn_bounds: Optional[List[Optional[Callable]]]=None
    ):
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

        self.ref_path = ref_path
        self.save_path = save_path

        assert all([isinstance(el, Statistic) for el in additional_stats])
        self.stats = {
            'mi': MI(),
            'condh': Condh(),
            'corr': Corr(),
        }
        self.stats.update(additional_stats)

        if fn_bounds is None:
            fn_bounds = [None] * len(target_names)
        self.fn_bounds = fn_bounds

# Writing access

    def create(
        self,
        targets: Sequence[Union[str, Sequence[str]]]
    ):
        """
        Create the statistics directory if not exists as well as the JSON files for targets in `targets`.
        """
        # TODO
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        
        for tar in targets:
            path = self._targets_to_filename(tar)
            if not os.path.isfile(path):
                with open(path, 'w', encoding="utf-8") as f:
                    json.dump([], f, ensure_ascii=False, indent=4)

    def remove(
        self,
        targets: Optional[Sequence[Union[str, Sequence[str]]]],
        variables: Optional[Sequence[Union[str, Sequence[str]]]]
    ) -> None:
        """
        If `targets` and `variables` are both None, remove the whole directory if exists.
        If `targets` and `variables` are both not None, only remove the corresponding pickle file.
        Else raise an error.
        """
        if targets is None and variables is None:
            if os.path.exists(self.save_path):
                shutil.rmtree(self.save_path)

        if (targets is None) ^ (variables is None):
            raise ValueError("targets and variables must be simultaneously None or not None")

        path = self._get_filename(targets, variables)
        if os.path.isfile(path):
            os.remove(path)

    def delete_stats(
        self,
        targs: Union[str, List[str]],
        stats: Union[str, List[str]]
    ) -> None:
        # TODO
        pass

    def update(
        self,
        inputs_dict: Dict[str, Any],
    ) -> None:
        self.store(
            inputs_dict,
            overwrite=False
        )

    def overwrite(
        self,
        inputs_dict: Dict[str, Any],
    ) -> None:
        self.store(
            inputs_dict,
            overwrite=True
        )

    def store(
        self,
        inputs_dict: Dict[str, Any],
        overwrite: bool=False
    ):
        """
        Inputs_dict:
        - TODO
        """

        # Get filename from targs and vars
        # Load .pickle file
        # Access the right regime
        # Access the right stat

        # Overwrite : écrase le fichier complet avec tous ses régimes
        # Not overwrite : écrase seulement les régimes indiqués

        # TODO


    # Reading access

    def read(
        self,
        targs: Union[str, Sequence[str]],
        vars: Union[str, Sequence[str]],
        regs: Union[str, Sequence[str]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        if isinstance(targs, str):
            targs = [targs]
        if isinstance(targs, tuple):
            targs = list(targs)

        if isinstance(vars, str):
            vars = [vars]
        if isinstance(vars, tuple):
            vars = list(vars)

        if isinstance(regs, str):
            regs = [regs]
        if isinstance(regs, tuple):
            regs = list(regs)

        # TODO
    
    def _get_filename(
        self,
        targets: Union[str, Sequence[str]],
        variables: Union[str, Sequence[str]]
    ) -> str:
        if isinstance(targets, str):
            targets = [targets]
        if isinstance(variables, str):
            targets = [variables]
        _targets = "_".join(sorted(targets)) + "__" + "_".join(sorted(variables))
        return os.path.join(self.save_path, _targets + '.npz')

    def _check_inputs(
        self,
        inputs_dict: Dict[str, Any],
        ref_dict: Dict[str, Any]
    ) -> None:
        # TODO
        # raise ValueError("Inputs must match the reference file")
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

        # Remove pixels out of the targets ranges (including NaNs)
        for rgs in ranges:
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

    # Others

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

    def __str__(self):
        return f"ContinuousHandler (path: {self.path})"

