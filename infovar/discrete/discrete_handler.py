import os
import shutil
import json
import yaml
import itertools as itt
from typing import List, Dict, Any, Tuple, Union, Optional, Sequence, Callable
from time import time

import numpy as np
from tqdm import tqdm

from ..stats.statistics import Statistic, MI, Condh, Corr

__all__ = [
    "DiscreteHandler"
]

class DiscreteHandler:

    variables: np.ndarray
    targets: np.ndarray
    variable_names: List[str]
    target_names: List[str]
    ref_path: str
    save_path: str
    additional_stats: Dict[str, Callable]

    def __init__(
        self,
        variables: np.ndarray,
        targets: np.ndarray,
        variable_names: Union[List[str], str],
        target_names: Union[List[str], str],
        ref_path: str,
        save_path: str,
        additional_stats: Dict[str, Callable]={},
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

    # Writing access

    def create(
        self,
        targets: Sequence[Union[str, Sequence[str]]]
    ):
        """
        Create the statistics directory if not exists as well as the JSON files for targets in `targets`.
        """
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        
        for tar in targets:
            if isinstance(tar, str):
                path = self._targets_to_filename(tar)
            if not os.path.isfile(path):
                with open(path, 'w', encoding="utf-8") as f:
                    json.dump([], f, ensure_ascii=False, indent=4)

    def remove(
        self,
        targets: Optional[Sequence[Union[str, Sequence[str]]]]=None
    ):
        """
        If `targets` is None, remove the statistics directory if exists.
        If `targets` is not None, only remove the corresponding JSON files.
        """
        if targets is None:
            if os.path.exists(self.save_path):
                shutil.rmtree(self.save_path)

        for tar in targets:
            path = self._targets_to_filename(tar)
            if not os.path.isfile(path):
                with open(path, 'w', encoding="utf-8") as f:
                    json.dump([], f, ensure_ascii=False, indent=4)

    def update(
        self,
        inputs_dict: Dict[str, Any],
    ):
        self.store(
            inputs_dict,
            overwrite=False
        )

    def overwrite(
        self,
        inputs_dict: Dict[str, Any],
    ):
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
        # Process ranges (cartesian product, concatenation, etc)
        ranges = self._list_ranges(
            inputs_dict["ranges"]
        )

        # Load the reference 
        with open(self.ref_path, 'r') as f:
            ref_dict = yaml.safe_load(f)  # results is a list of dicts

        # Checks that the file is in the expected format.
        # In particular, checks that variables, targets and ranges are in the reference file.
        self._check_inputs(
            inputs_dict,
            ref_dict,
        )

        # Create directory of not exists
        self.create(inputs_dict["targets"])

        # Targets loop
        for targs in inputs_dict['targets']:

            if isinstance(targs, str):
                targs = [targs]

            # Load existing data
            path = self._targets_to_filename(targs)
            with open(path, 'r') as f:
                results = json.load(f)  # results is a list of dicts

            # targs_idx = [self.target_names.index(v) for v in targs]
            # _targs = np.column_stack([
            #     self.targets[i] for i in targs_idx
            # ])

            # Variables loop
            pbar = tqdm(
                self._comb(inputs_dict['variables'], inputs_dict['n_variables']),
                desc=f"{targs}"
            )
            for vars in pbar:
                            
                vars = sorted(vars)
                pbar.set_postfix({'vars': f"{vars}"})

                # We check if the combination of variables already exists
                index_vars = self._index_of(
                    results, value=vars, key="vars"
                )

                # We create an entry if it doesn't exists
                if index_vars is None:
                    results.append({
                        "vars": vars,
                        "stats": [],
                    })
                    index_vars = -1

                # vars_idx = [self.variable_names.index(v) for v in vars]
                # _vars = np.column_stack([
                #     self.variables[i] for i in vars_idx
                # ])

                # Ranges loop
                for rgs in ranges:

                    # We check if ranges already exists
                    index_ranges = self._index_of_ranges(
                        results[index_vars]["stats"], value=rgs, key="ranges"
                    )

                    # We keep the existing results if `overwrite`` is False
                    if not overwrite and index_ranges is not None:
                        continue

                    # We create an entry if it doesn't exists
                    if index_ranges is None:
                        results[index_vars]["stats"].append({
                            "ranges": rgs,
                            "stats": {},
                        })
                        index_ranges = -1

                    # Ranges restriction
                    _rgs = {t: ref_dict["ranges"][t][r] for t, r in rgs.items()}

                    _X, _Y = self._filter_data(
                        vars, targs, _rgs
                    )
                    samples = _Y.shape[0]

                    for stat in inputs_dict["statistics"]:

                        operator = self.stats[stat] # Callable(ndarray, ndarray) -> float

                        # Simple computation
                        # TODO définir dans les inputs un nombre minimal d'échantillons
                        value = None
                        try:
                            start = time()
                            value = operator(_X, _Y)
                            end = time()
                            res = {
                                f'{stat}': value,
                                f'{stat}-time': end-start
                            }
                        except:
                            res = {
                                f'{stat}': None,
                                f'{stat}-time': None
                            }

                        # Bootstrapping
                        if value is not None and inputs_dict["bootstrapping"][stat] is not None:
                            boots = []
                            for _ in range(inputs_dict["bootstrapping"][stat]):
                                idx = np.random.choice(samples, samples, replace=True)
                                boots.append(operator(_X[idx], _Y[idx]))
                                res.update({
                                f'{stat}-boot': boots
                            })
                        else:
                            res.update({
                                f'{stat}-boot': None
                            })

                        # Store statistics
                        results[index_vars]["stats"][index_ranges]["stats"].update({
                            **res,
                            "samples": samples,
                        })

                # Save results (update for each variables iteration)
                with open(self._targets_to_filename(targs), 'w', encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=4)


    # Reading access

    def read(
        self,
        targs: Union[str, Sequence[str]],
        vars: Union[str, Sequence[str]],
        ranges: Dict[str, str],
    ):
        if isinstance(vars, str):
            vars = [vars]
        if isinstance(vars, tuple):
            vars = list(vars)

        # TODO pour les targets, il faut trier par ordre alphabétique

        # Load data
        path = self._targets_to_filename(targs)
        if not os.path.exists(path):
            raise FileNotFoundError(f"File {path} not exists yet.")
        with open(path, 'r') as f:
            data = json.load(f)  # results is a list of dicts

        # Find the good set of variables
        vars = set(vars)
        found = False
        for item in data:
            if set(item["vars"]) == vars:
                data = item["stats"]
                found = True
                break
        if not found:
            raise ValueError(f"Variables {vars} doesn't exist in data")

        # Find the good ranges
        found = False
        for item in data:
            if item["ranges"] == ranges:
                data = item["stats"]
                found = True
                break
        if not found:
            raise ValueError(f"Ranges of data {ranges} doesn't exist in data")

        return data


    @staticmethod
    def _index_of(
        ls: Sequence[Dict[str, Sequence]],
        value: Sequence,
        key: str
    ) -> Optional[int]:
        """
        Order of element and case are important.
        """
        try:
            index = [tuple(item[key]) for item in ls].index(tuple(value))
        except ValueError:
            index = None
        return index

    @staticmethod
    def _index_of_ranges(
        ls: Sequence[Dict[str, Dict]],
        value: Sequence,
        key: str
    ) -> Optional[int]:
        """
        Order of element and case are important.
        """
        try:
            index = [item[key] for item in ls].index(value)
        except ValueError:
            index = None
        return index

    @staticmethod
    def _comb(
        ls: Sequence,
        repeats: Union[int, Sequence[int]]
    ) -> List[Sequence]:
        res = []
        for r in repeats:
            res += list(itt.combinations(ls, r))
        return res
    
    def _targets_to_filename(
        self,
        targets: Union[str, Sequence[str]]
    ) -> str:
        if isinstance(targets, str):
            targets = [targets]
        _targets = "_".join(sorted(targets))
        return os.path.join(self.save_path, _targets + '.json')

    @staticmethod
    def _list_ranges(
        ls: List[Dict[str, Union[str, List[str]]]]
    ) -> List[Dict[str, str]]:
        full = []
        for el in ls:
            _el = el.copy()
            for key in _el:
                if not isinstance(_el[key], List):
                    _el[key] = [_el[key]]
            keys = list(_el.keys())
            for vals in itt.product(*list(_el.values())):
                full.append({k: v for k, v in zip(keys, vals)})
        return full

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
            if ranges[rgs] is None:
                pass
            else:
                a, b = ranges[rgs]
                i = self.target_names.index(rgs)
                if a is not None:
                    filt &= a <= self.targets[:, i]
                if b is not None:
                    filt &= self.targets[:, i] <= b

        _vars = _vars[filt]
        _targs = _targs[filt]
                
        # # Replace observables NaNs by a Gaussian noise TODO
        # if X_noise is not None and X_noise.size > 1:
        #     X_noise = np.where(
        #         np.isnan(X_noise),
        #         np.nanstd(X_noise),
        #         X_noise
        #     )
        # if X is not None:
        #     X = np.where(
        #         np.isnan(X),
        #         np.log10(np.abs(np.random.normal(0, X_noise))),
        #         X
        #     ) # log10, additive noise mimic real noise dynamic
            
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
        return f"DiscreteHandler (path: {self.path})"