import os
import shutil
import json
import itertools as itt
from typing import List, Dict, Any, Tuple, Union, Optional, Sequence, Callable, Iterable
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

    save_path: str

    getter: Callable[[List[str], List[str], Dict[str, Tuple[float, float]]], Tuple[np.ndarray, np.ndarray]]

    stats: Dict[str, Callable]
    resamplings: Dict[str, Resampling]={}

    restrictions: Optional[Dict[str, Dict]]=None


    # Setter

    def set_restrictions(self, d: Dict[str, Dict]) -> None:
        self.restrictions = d


    # General

    def get_filename(
        self,
        y_names: Union[str, Sequence[str]]
    ) -> str:
        if isinstance(y_names, str):
            y_names = [y_names]
        _y_names = "_".join(sorted(y_names))
        return os.path.join(self.save_path, _y_names + '.json')


    # Writing access

    def create(
        self,
        y_names: Union[str, Sequence[str]]
    ):
        """
        Create the statistics directory if not exists as well as the JSON files for features in `y_names`.
        """
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        
        path = self.get_filename(y_names)
        if not os.path.isfile(path):
            with open(path, 'w', encoding="utf-8") as f:
                json.dump([], f, ensure_ascii=False, indent=4)

    def remove(
        self,
        y_names: Optional[Sequence[Union[str, Sequence[str]]]]
    ) -> None:
        """
        If `y_names` is None, remove the whole directory if exists.
        If `y_names` is not None, only remove the corresponding JSON files.
        """
        if not os.path.isdir(self.save_path):
            raise FileNotFoundError(f"Save directory {self.save_path} does not exist.")


        if y_names is None:
            if any([not file.endswith((".json", ".pickle", ".pkl")) for file in os.listdir(self.save_path)]):
                raise PermissionError(f"Save directory {self.save_path} contains files or directories that are not generated with handlers. You should remove it by hand.")
            shutil.rmtree(self.save_path)
            return

        for tar in y_names:
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

        with open(path + ".tmp", 'w', encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False, indent=4)
        os.rename(path + ".tmp", path)

    def store(
        self,
        x_names: Union[str, List[str], Iterable[List[str]]],
        y_names: Union[str, List[str]],
        inputs_dict: Dict[str, Any],
        overwrite: bool=False,
        iterable_x: bool=False,
        save_every: int=1,
        progress_bar: bool=True,
        total_iter: int=None,
        raise_error: bool=True
    ) -> None:
        """
        Inputs_dict:
        - TODO
        """

        # Checks that the file is in the expected format.
        self._check_inputs(
            inputs_dict,
        )

        if isinstance(y_names, str):
            y_names = [y_names]

        y_names = sorted(y_names)

        # Create directory of not exists
        self.create(y_names)

        # Load existing data
        path = self.get_filename(y_names)
        with open(path, 'r') as f:
            results = json.load(f)  # results is a list of dicts

        # Variables loop
        if iterable_x:
            assert isinstance(x_names, Iterable) and not isinstance(x_names, str)
        else:
            x_names = [x_names]

        pbar = tqdm(
            x_names,
            desc=str(y_names).replace("'", ""),
            total=total_iter,
            disable=not progress_bar
        )
        for it, _x_names in enumerate(pbar, 1):
            if isinstance(_x_names, str):
                _x_names = [_x_names]    

            _lines = list(set(_x_names))
            pbar.set_postfix({'x': str(_lines).replace("'", "")})

            _x_names = list(set(_x_names))            
            _x_names = sorted(_x_names)

            # We check if the combination of variables already exists
            index_x = self._index_of(
                results, value=_x_names, key="x_names"
            )

            # We create an entry if it doesn't exists
            if index_x is None:
                results.append({
                    "x_names": _x_names,
                    "stats": [],
                })
                index_x = -1

            # Ranges loop
            for restr in inputs_dict["restrictions"]: # TODO: "restrictions" field can be None

                # We check if ranges already exists
                index_ranges = self._index_of_ranges(
                    results[index_x]["stats"], value=restr, key="restriction"
                )

                # We create an entry if it doesn't exists
                if index_ranges is None:
                    results[index_x]["stats"].append({
                        "restriction": restr,
                        "stats": {},
                    })
                    index_ranges = -1

                # Ranges restriction
                if self.restrictions is not None and restr is not None:
                    restrict_dict = self.restrictions[restr]
                elif self.restrictions is not None:
                    raise ValueError(f"self.restriction must not be None when the restriction asked is not None (here {restr}). Consider using set_restrictions to load the dictionnary.")
                else:
                    restrict_dict = {}

                _X, _Y = self.getter(
                    _x_names, y_names, restrict_dict, inputs_dict.get('max_samples')
                )

                entry = results[index_x]["stats"][index_ranges]["stats"]
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
                        inputs_dict, entry[stat],
                        raise_error=raise_error
                    ) # Modify `entry` in-place
                
                samples = _Y.shape[0]
                prev_samples = entry.get("samples")
                if prev_samples is not None and samples != samples:
                    raise ValueError("Old and new number of samples are differents.")

                entry.update({
                    "samples": samples
                })

            # Save results
            
            if it % save_every == 0:
                path = self.get_filename(y_names)
                with open(path + ".tmp", 'w', encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=4)
                os.rename(path + ".tmp", path)

        # Final save
        
        path = self.get_filename(y_names)
        with open(path + ".tmp", 'w', encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        os.rename(path + ".tmp", path)

    def _compute_stat(
        self,
        X: np.ndarray, Y: np.ndarray,
        operator: Statistic, stat: str,
        inputs_dict: Dict[str, Any], entry: Dict[str, Any],
        raise_error: bool=True
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
        except Exception as e:
            if raise_error:
                raise e
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
        except Exception as e:
            if raise_error:
                raise e
            entry["std"] = None
            return

        try:
            std = self.resamplings[name].compute_sigma(X, Y, operator, **args)
        except Exception as e:
            if raise_error:
                raise e
            std = None
        entry.update({
            "std": std
        })

        return entry

    # Reading access

    def _get_variables_content(
        self,
        x_names: List[str],
        data: List[Dict[str, Any]]
    ) -> Optional[List[Dict]]:
        _x_names = set(x_names)
        for _item in data:
            if set(_item["x_names"]) == _x_names:
                item = _item["stats"]
                return item
        return None
    
    def _get_restriction_content(
        self,
        restr: str,
        data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        for _item in data:
            if _item["restriction"] == restr:
                item = _item["stats"]
                return item
        return None

    def read(
        self,
        x_names: Union[str, List[str], Iterable[List[str]]],
        y_names: Union[str, List[str]],
        restr: str,
        iterable_x: bool=False,
        default: Any="raise"
    ):
        if isinstance(x_names, str):
            x_names = [x_names]

        if isinstance(y_names, str):
            y_names = [y_names]
        assert isinstance(y_names, List)

        # Load data
        path = self.get_filename(y_names)
        if not os.path.exists(path):
            raise FileNotFoundError(f"File {path} not exists yet.")
        with open(path, 'r') as f:
            data = json.load(f)  # results is a list of dicts

        if not iterable_x:
            x_names = [x_names]

        item_list = []
        for _x_names in x_names:
            if isinstance(_x_names, str):
                _x_names = [_x_names]

            # Find the good set of variables
            item = self._get_variables_content(_x_names, data)
            if item is None:
                if default == "raise":
                    msg = f"Variables {_x_names} doesn't exist in data."
                    if any([isinstance(el, (List, Tuple)) for el in _x_names]):
                        msg += f" It seems that you provide an Iterable, did you missed setting the `iterable_x` flag?"
                    raise ValueError(msg)
                item_list.append(default)
                continue

            # Find the good restriction
            item = self._get_restriction_content(restr, item)
            if item is None:
                if default == "raise":
                    raise ValueError(f"Restriction of data {restr} doesn't exist in data")
                item = default

            # Store value
            item_list.append(item)

        if iterable_x:
            return item_list
        return item_list[0]
    
    def get_available_targets(
        self
    ) -> List[List[str]]:
        """
        TODO
        """
        filenames = [
            f.replace(".json", "") for f in os.listdir(self.save_path) if f.endswith(".json")
        ]
        return [
            f.split("_") for f in filenames
        ]

    def get_available_variables(
        self,
        targets: Union[str, List[str]],
    ) -> List[List[str]]:
        """
        TODO
        """
        # Load data
        path = self.get_filename(targets)
        if not os.path.exists(path):
            raise FileNotFoundError(f"File {path} not exists yet.")
        with open(path, 'r') as f:
            data = json.load(f)  # results is a list of dicts

        return [item["x_names"] for item in data]
    
    def get_available_restrictions(
        self,
        targets: Union[str, List[str]],
        variables: Union[str, List[str]]
    ) -> List[str]:
        """
        TODO
        """
        if isinstance(targets, str):
            targets = [targets]
        if isinstance(variables, str):
            variables = [variables]

        # Load data
        path = self.get_filename(targets)
        if not os.path.exists(path):
            raise FileNotFoundError(f"File {path} not exists yet.")
        with open(path, 'r') as f:
            data = json.load(f)  # results is a list of dicts

        # Get variables content
        content = self._get_variables_content(variables, data)

        return [item["restriction"] for item in content]

    def get_available_stats(
        self,
        targets: Union[str, List[str]],
        variables: Union[str, List[str]],
        restriction: str
    ) -> List[str]:
        if isinstance(targets, str):
            targets = [targets]
        if isinstance(variables, str):
            variables = [variables]

        # Load data
        path = self.get_filename(targets)
        if not os.path.exists(path):
            raise FileNotFoundError(f"File {path} not exists yet.")
        with open(path, 'r') as f:
            data = json.load(f)  # results is a list of dicts

        # Get variables content
        content = self._get_variables_content(variables, data)
        if content is None:
            raise ValueError(f"Invalid variables {variables} for targets {targets}")

        for entry in content:
            if entry["restriction"] == restriction:
                return list(entry["stats"].keys())
        
        raise ValueError(f"Invalid restriction {restriction} for targets {targets} and variables {variables}")

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

    def _check_inputs(
        self,
        inputs_dict: Dict[str, Any],
    ) -> None:

        # TODO
        pass
                

    # Display

    def __str__(self):
        return f"DiscreteHandler"
