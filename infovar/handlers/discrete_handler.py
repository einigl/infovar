import os
import shutil
import json
import yaml
import itertools as itt
from typing import List, Dict, Any, Tuple, Union, Optional, Sequence, Callable
from time import time

import numpy as np
from tqdm import tqdm

from ..stats.statistics import Statistic
from ..stats.resampling import Resampling

from .handler import Handler

__all__ = [
    "DiscreteHandler"
]


class DiscreteHandler(Handler):

    ref_path: str
    save_path: str

    variables: Optional[np.ndarray]
    targets: Optional[np.ndarray]
    variable_names: Optional[List[str]]
    target_names: Optional[List[str]]

    additional_stats: Dict[str, Callable]
    additional_resamplings: Dict[str, Resampling]={},

    fn_bounds: Optional[List[Optional[Callable]]]
    inv_fn_bounds: Optional[List[Optional[Callable]]]


    # General

    def get_filename(
        self,
        targets: Union[str, Sequence[str]]
    ) -> str:
        if isinstance(targets, str):
            targets = [targets]
        _targets = "_".join(sorted(targets))
        return os.path.join(self.save_path, _targets + '.json')


    # Writing access

    def create(
        self,
        targets: Union[str, Sequence[str]]
    ):
        """
        Create the statistics directory if not exists as well as the JSON files for targets in `targets`.
        """
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        
        path = self.get_filename(targets)
        if not os.path.isfile(path):
            with open(path, 'w', encoding="utf-8") as f:
                json.dump([], f, ensure_ascii=False, indent=4)

    def remove(
        self,
        targets: Optional[Sequence[Union[str, Sequence[str]]]]
    ) -> None:
        """
        If `targets` is None, remove the whole directory if exists.
        If `targets` is not None, only remove the corresponding JSON files.
        """
        if targets is None:
            if os.path.exists(self.save_path):
                shutil.rmtree(self.save_path)

        for tar in targets:
            path = self.get_filename(tar)
            if os.path.isfile(path):
                os.remove(path)

    def delete_stats(
        self,
        targs: Union[str, List[str]],
        stats: Union[str, List[str]]
    ) -> None:
        path = self.get_filename(targs)
        with open(path, 'r') as f:
            d = json.load(f)  # results is a list of dicts

        if isinstance(stats, str):
            stats = [stats]

        for stat in stats:
            for item1 in d:
                for item2 in item1["stats"]:
                    item3 = item2["stats"]
                    for key in [f'{stat}', f'{stat}-time', f'{stat}-boot']:
                        item3.pop(key, None)

        with open(path, 'w', encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False, indent=4)

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

        # Targets loop
        for targs in inputs_dict['targets']:

            # Create directory of not exists
            self.create(targs)

            if isinstance(targs, str):
                targs = [targs]

            # Load existing data
            path = self.get_filename(targs)
            with open(path, 'r') as f:
                results = json.load(f)  # results is a list of dicts

            # Variables loop
            pbar = tqdm(
                self._comb(inputs_dict['variables'], inputs_dict['n_variables']),
                desc=str(targs).replace("'", "")
            )
            for vars in pbar:
                            
                vars = sorted(vars)
                pbar.set_postfix({'vars': str(vars).replace("'", "")})

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

                # Ranges loop
                for rgs in ranges:

                    # We check if ranges already exists
                    index_ranges = self._index_of_ranges(
                        results[index_vars]["stats"], value=rgs, key="ranges"
                    )

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

                    entry = results[index_vars]["stats"][index_ranges]["stats"]
                    for stat in inputs_dict["statistics"]:
                        if overwrite and stat in entry:
                            entry.pop(stat, None)
                    if set(entry.keys()) <= {"samples"}: # If no keys or only "samples"
                        entry.pop("samples", None)

                    for stat in inputs_dict["statistics"]:

                        if stat in entry:
                            continue
                        entry.update({stat: {}})

                        operator = self.stats[stat]

                        self._compute_stat(
                            _X, _Y,
                            operator, stat,
                            inputs_dict, entry[stat]
                        ) # Modify `entry` in-place
                    
                    samples = _Y.shape[0]
                    prev_samples = entry.get("samples")
                    if prev_samples is not None and samples != samples:
                        raise ValueError("Old and new number of samples are differents.")

                    entry.update({
                        "samples": samples
                    })

                # Save results (update for each variables iteration)
                with open(self.get_filename(targs), 'w', encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=4)

    def _compute_stat(
        self,
        X: np.ndarray, Y: np.ndarray,
        operator: Statistic, stat: str,
        inputs_dict: Dict[str, Any], entry: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Modifies in-place the `entry` dictionnary.
        Adds or changes the three following entries of `entry`:
        - {stat}: computed only if current value is None.
            If error, the default value is None.
        - {stat}-time: computed only if new value of {stat} is not None.
        - {stat}-boot: computed only if new value of {stat} is not None
            and if `inputs_dict["bootstrapping"][stat]` is not None.
            If current value is None, then the entry is replaced by the new array.
            If current value is not None, then we compute as many values as necessary to reach
            `inputs_dict["bootstrapping"][stat]` total values.

        Note: if the available number of samples is lower than `inputs_dict["min_samples"]`,
        then all three entries are set to None.
        """
        # Samples
        samples = Y.shape[0]
        if samples <= inputs_dict["min_samples"]:
            return {
                "value": None,
                "time": None,
                "std": None
            }

        # Simple computation
        try:
            start = time()
            value = operator(X, Y)
            end = time()
            entry.update({
                "value": value,
                "time": end-start,
            })
        except:
            entry.update({
                "value": None,
                "time": None,
                "std": None
            })
            return entry

        # Uncertainty
        try:
            d = inputs_dict["uncertainty"][stat]
            name, args = d["name"], d["args"]
        except:
            entry["std"] = None
            return

        try:
            std = self.resamplings[name].compute_sigma(X, Y, operator, **args)
        except:
            std = None
        entry.update({
            "std": std
        })

        return entry

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
        path = self.get_filename(targs)
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
    
    def get_available_targets(
        self
    ):
        raise NotImplementedError("")

    def get_available_variables(
        self,
        targets: Union[str, List[str]],
    ):
        raise NotImplementedError("")

    def get_available_stats(
        self,
        targets: Union[str, List[str]],
        variables: Union[str, List[str]],
        windows: Union[str, List[str]]
    ):
        raise NotImplementedError("")

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
        # Reference file

        key = "variables"
        assert isinstance(ref_dict[key], List)
        assert all([isinstance(el, (str, List)) for el in ref_dict[key]])

        key = "targets"
        assert isinstance(ref_dict[key], List)
        assert all([isinstance(el, (str, List)) for el in ref_dict[key]])

        key = "ranges"
        # TODO

        # Inputs file

        mandatory_keys = [
            "variables",
            "targets",
            "statistics",
        ]
        for key in mandatory_keys:
            assert key in inputs_dict

        optional_keys = [
            "n_variables",
            "ranges",
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
        assert all([k in ref_dict[key] for k in inputs_dict[key]])
        #

        key = "targets"
        assert isinstance(inputs_dict[key], List)
        assert all([isinstance(el, (str, List)) for el in inputs_dict[key]])
        #
        assert all([k in ref_dict[key] for k in inputs_dict[key] if isinstance(k, str)])
        #

        key = "n_variables"
        if key not in inputs_dict:
            inputs_dict[key] = [1]
        if isinstance(inputs_dict[key], int):
            inputs_dict[key] = [inputs_dict[key]]
        assert isinstance(inputs_dict[key], List)
        assert all([isinstance(el, int) for el in inputs_dict[key]])

        key = "ranges"
        if key not in inputs_dict:
            inputs_dict[key] = None
        assert isinstance(inputs_dict[key], List)
        assert all([isinstance(el, Dict) for el in inputs_dict[key]])
        #
        # TODO
        #

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
        

        return inputs_dict


    # Display

    def __str__(self):
        return f"DiscreteHandler"
