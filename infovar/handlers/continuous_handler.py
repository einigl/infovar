import os
import shutil
import pickle
import yaml
import itertools as itt
from typing import List, Dict, Any, Tuple, Union, Optional, Sequence, Callable

import numpy as np
from tqdm import tqdm

from ..stats.statistics import Statistic
from ..stats.resampling import Resampling

from .handler import Handler


__all__ = [
    "ContinuousHandler"
]

class ContinuousHandler(Handler):

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


    # Writing access

    def create(
        self,
        targets: Union[str, Sequence[str]],
        variables: Union[str, Sequence[str]]
    ):
        """
        Create the statistics directory if not exists as well as the JSON files for targets in `targets`.
        """
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        
        path = self.get_filename(targets, variables)
        if not os.path.isfile(path):
            with open(path, 'wb') as f:
                pickle.dump([], f)

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

        path = self.get_filename(targets, variables)
        if os.path.isfile(path):
            os.remove(path)

    def delete_stats(
        self,
        targs: Union[str, List[str]],
        stats: Union[str, List[str]]
    ) -> None:
        path = self._targets_to_filename(targs)
        with open(path, 'rb') as f:
            d = pickle.load(f)  # results is a list of dicts

        if isinstance(stats, str):
            stats = [stats]

        for stat in stats:
            for item1 in d:
                item2 = item1["stats"]
                for key in [f'{stat}-coord', f'{stat}-data']:
                    item2.pop(key, None)

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

        # Load the reference 
        with open(self.ref_path, 'r') as f:
            ref_dict = yaml.safe_load(f)  # results is a list of dicts

        # Checks that the file is in the expected format.
        # In particular, checks that variables, targets and ranges are in the reference file.
        self._check_inputs(
            inputs_dict,
            ref_dict,
        )

        # Targets loop
        for targs in inputs_dict['targets']:
            if isinstance(targs, str):
                targs = [targs]
            targs = sorted(targs)

            # Variables loop
            pbar = tqdm(
                inputs_dict['variables'],
                desc=f"{targs}"
            )
            for vars in inputs_dict['variables']:
                if isinstance(vars, str):
                    vars = [vars]
                vars = sorted(vars)
                pbar.set_postfix({'vars': f"{vars}"})

                # Create directory of not exists
                self.create(targs, vars)

                if isinstance(targs, str):
                    targs = [targs]
                if isinstance(vars, str):
                    vars = [vars]

                # Load existing data
                path = self.get_filename(targs, vars)
                with open(path, 'rb') as f:
                    results = pickle.load(f)  # results is a list of dicts

                # Windows loop
                for wins in inputs_dict['windows']:

                    wins = wins.copy()
                    for key in ['targets', 'length', 'points']:
                        if not isinstance(wins[key], List):
                            wins[key] = [wins[key]]

                    # We check if the data already exists
                    index_windows = self._index_of_data(
                        results, value=wins['targets'], key="windows"
                    )

                    # We create an entry if it doesn't exists
                    if index_windows is None:
                        results.append({
                            "windows": wins['targets'],
                            "stats": {},
                        })
                        index_windows = -1

                    entry = results[index_windows]["stats"]
                    for stat in inputs_dict["statistics"]:

                        # We reset the existing results if `overwrite` is True
                        if overwrite :
                            entry.pop(f"{stat}-coord", None)
                            entry.pop(f"{stat}-data", None)
                            entry.pop("samples", None)

                        operator = self.stats[stat] # Callable(ndarray, ndarray) -> float

                        self._compute_stat(
                            targs, vars, wins,
                            operator, stat,
                            inputs_dict, entry
                        ) # Modify `entry` in-place

                    # Save results (update for each ranges iteration)
                    with open(self.get_filename(targs, vars), 'wb') as f:
                        pickle.dump(results, f)

    def _compute_stat(
        self,
        targs: List[str], vars: List[str], wins: Dict[str, Any],
        operator: Statistic, stat: str,
        inputs_dict: Dict[str, Any], entry: Dict[str, Any]
    ) -> Dict[str, Any]:
        
        coords = []
        bounds = []
        for targ, perc, pts in zip(wins['targets'], wins['length'], wins['points']):
            arr = self.targets[:, self.target_names.index(targ)]
            winsize = (perc/100 if perc>=1 else perc) * np.ptp(arr)
            low, high = np.min(arr), np.max(arr)

            inv_fn = self.inv_fn_bounds[self.target_names.index(targ)]

            xticks = np.linspace(low+winsize/2, high-winsize/2, pts)
            bounds.append( (inv_fn(xticks-winsize/2), inv_fn(xticks+winsize/2)) )
            coords.append(inv_fn(xticks))

        data = np.zeros(wins['points'])
        samples = np.zeros(wins['points'])

        try:
            d = inputs_dict["uncertainty"][stat]
            name, args = d["name"], d["args"]
        except:
            if f'{stat}-std' not in entry:
                entry[f'{stat}-std'] = None
            return
        
        if stat in inputs_dict["uncertainty"] and "name" in inputs_dict["uncertainty"][stat]: 
            std = np.zeros(wins['points'])
            name_resampling = inputs_dict["uncertainty"][stat]["name"]
            args = inputs_dict["uncertainty"][stat]["args"]
        else:
            std = None

        for ii in tqdm(
            itt.product(*[range(n) for n in wins['points']]),
            total=np.prod(wins['points'])
        ):
            ranges = {key: [b[0][i], b[1][i]] for i, b, key in zip(ii, bounds, wins['targets'])}
            _X, _Y = self._filter_data(
                vars, targs, ranges
            )

            # Samples
            n = _Y.shape[0]
            samples[ii] = n

            # Computation of value
            if n >= inputs_dict['min_samples']:
                try:
                    data[ii] = operator(_Y, _X)
                except:
                    data[ii] = np.nan
            else:
                data[ii] = np.nan

            if std is None:
                continue

            # Computation of estimator variance
            if n >= inputs_dict['min_samples']:
                try:
                    std[ii] = self.resamplings[name_resampling].compute_sigma(_X, _Y, operator, **args)
                except:
                    std[ii] = np.nan
            else:
                std[ii] = np.nan

        entry[f'{stat}-coords'] = tuple(coords)
        entry[f'{stat}-data'] = data
        entry[f'{stat}-std'] = std
        entry['samples'] = samples
        return entry


    # Reading access

    def read(
        self,
        targs: Union[str, Sequence[str]],
        vars: Union[str, Sequence[str]],
        wins_targs: Union[str, Sequence[str]]
    ) -> Dict[str, Any]:
        # Formatting
        if isinstance(targs, str):
            targs = [targs]
        if isinstance(vars, str):
            vars = [vars]
        if isinstance(wins_targs, str):
            wins_targs = [wins_targs]

        # Load data
        path = self.get_filename(targs, vars)
        if not os.path.exists(path):
            raise FileNotFoundError(f"File {path} not exists yet.")
        with open(path, 'rb') as f:
            data = pickle.load(f)  # data is a list of dicts

        # Find the good windows
        names = set(wins_targs)
        found = False
        for item in data:
            if set(item["windows"]) == names:
                data = item["stats"]
                found = True
                break
        if not found:
            raise ValueError(f"Windows {vars} doesn't exist in data")

        return data

    @staticmethod
    def _index_of_data(
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
    
    def get_filename(
        self,
        targets: Union[str, Sequence[str]],
        variables: Union[str, Sequence[str]]
    ) -> str:
        if isinstance(targets, str):
            targets = [targets]
        if isinstance(variables, str):
            targets = [variables]
        _targets = "_".join(sorted(targets)) + "__" + "_".join(sorted(variables))
        return os.path.join(self.save_path, _targets + '.pickle')

    def _check_inputs(
        self,
        inputs_dict: Dict[str, Any],
        ref_dict: Dict[str, Any]
    ) -> None:
        # Reference file

        key = "variables"
        assert isinstance(ref_dict[key], List)
        assert all([isinstance(el, (str, List)) for el in ref_dict[key]])

        key = "targets"
        assert isinstance(ref_dict[key], List)
        assert all([isinstance(el, (str, List)) for el in ref_dict[key]])


        # Inputs file

        mandatory_keys = [
            "variables",
            "targets",
            "windows",
            "statistics",
        ]
        for key in mandatory_keys:
            assert key in inputs_dict

        optional_keys = [
            "min_samples",
            "uncertainty"
        ]

        for key in inputs_dict:
            if key not in mandatory_keys:
                assert key in optional_keys

        key = "variables"
        assert isinstance(inputs_dict[key], List)
        assert all([isinstance(el, str) for el in inputs_dict[key]])
        #
        assert all([key in inputs_dict[key] in ref_dict[key]])
        #

        key = "targets"
        assert isinstance(inputs_dict[key], List)
        assert all([isinstance(el, (str, List)) for el in inputs_dict[key]])
        #
        assert all([key in inputs_dict[key] in ref_dict[key]])
        #

        key = "windows"
        assert isinstance(inputs_dict[key], List)
        assert all([isinstance(el, Dict) for el in inputs_dict[key]])
        # TODO

        key = "min_samples"
        if key not in inputs_dict:
            inputs_dict[key] = 0
        assert isinstance(inputs_dict[key], int)

        key = "statistics"
        if isinstance(inputs_dict[key], str):
            inputs_dict[key] = [inputs_dict[key]]
        assert isinstance(inputs_dict[key], List)
        assert all([isinstance(el, str) for el in inputs_dict[key]])

        key = "uncertainty"
        if key not in inputs_dict:
            inputs_dict[key] = {}
        assert isinstance(inputs_dict[key], Dict)
        assert all([isinstance(k, str) and isinstance(d, Dict)\
                    for k, d in inputs_dict[key].items()])
        assert all(["name" in d for d in inputs_dict[key].values()])
        for k in inputs_dict[key]:
            assert inputs_dict[key][k]["name"] in self.resamplings
            if "args" not in inputs_dict[key][k]:
                inputs_dict[key][k] = {}

        return inputs_dict


    # Display

    def __str__(self):
        return f"ContinuousHandler"
