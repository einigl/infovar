import json
import os
import shutil
from time import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
from tqdm import tqdm

from ..stats.resampling import Resampling
from ..stats.statistics import Statistic
from .handler import Handler

__all__ = ["DiscreteHandler"]


class DiscreteHandler(Handler):
    """
    Class for easily calculating, manipulating and saving calculations of statistical relationships between variables and targets according to predefined situations (restrictions). The term “discrete” means that the calculation is performed for a finite number of independent restrictions.
    """

    ext = ".json"   #: File extension
    restrictions: Optional[Dict[str, Dict]] = None  #: Dict of current restrictions
    filename_sep: str = "_"  #: Separator between targets

    # Setter

    def set_restrictions(self, d: Dict[str, Dict[str, Tuple[float, float]]]) -> None:
        """
        Set new restrictions, i.e., the constraints on one or more targets that reduce the number of data samples that can be used in the calculation.

        Parameters
        ----------
        d : Dict[str, Dict[str, Tuple[float, float]]]
            New restrictions.
        """
        if not isinstance(d, Dict):
            raise TypeError(f"Restriction dictionnary must be a Dict, not {type(d)}")
        t = self._check_dict_type(d, str, Dict)
        if t is not None:
            raise TypeError(
                f"{t} is not a valid entry ({type(t[0])}, {type(t[1])} instead of (str, Dict))"
            )

        for _d in d.values():
            t = self._check_dict_type(_d, str, List)
            if t is not None:
                raise TypeError(
                    f"{t} is not a valid entry ({type(t[0])}, {type(t[1])} instead of (str, Dict))"
                )

        self.restrictions = d

    # Saves

    def get_filename(self, y_names: Union[str, Sequence[str]]) -> str:
        """
        Builds a save filename from target names.

        Parameters
        ----------
        y_names : Union[str, Sequence[str]]
            Target names.

        Returns
        -------
        str
            Filename.
        """
        if isinstance(y_names, str):
            y_names = [y_names]
        _y_names = self.filename_sep.join(sorted(y_names))
        return os.path.join(self.save_path, _y_names + self.ext)

    def parse_filename(self, filename: str) -> Sequence[str]:
        """
        Identifies data names from save filename.

        Parameters
        ----------
        filename : str
            Save filename.

        Returns
        -------
        Sequence[str]
            Target names.
        """
        assert filename.endswith(self.ext)
        return filename.removesuffix(self.ext).split(self.filename_sep)

    # Writing access

    def create(self, y_names: Union[str, Sequence[str]]):
        """
        Create the statistics directory if not exists as well as the JSON files for features in `y_names`.

        Parameters
        ----------
        y_names : Union[str, Sequence[str]]
            Target names.
        """
        super().create()
        path = self.get_filename(y_names)
        if not os.path.isfile(path):
            with open(path, "w", encoding="utf-8") as f:
                json.dump([], f, ensure_ascii=False, indent=4)

    def remove(self, y_names: Optional[Sequence[Union[str, Sequence[str]]]]) -> None:
        """
        If `y_names` is None, remove the entire `self.save_path` directory.
        If `y_names` is not None, only remove the corresponding JSON file, if exists. If not, raise an error.
        """
        if not os.path.isdir(self.save_path):
            raise FileNotFoundError(f"Save directory {self.save_path} does not exist.")

        if y_names is None:
            super().remove()
            return

        for tar in y_names:
            path = self.get_filename(tar)
            if os.path.isfile(path):
                os.remove(path)

    def delete_stats(
        self,
        targs: Union[str, List[str]],
        stats: Union[str, List[str]],
        vars: Optional[Union[str, List[str]]] = None,
    ) -> None:
        """
        Removes stats `stats` for targets `targs` and variables `vars`.
        If `vars` is omitted, the stats are removed for any variable with the specified target.

        Parameters
        ----------
        targs : Union[str, List[str]]
            Targets.
        stats : Union[str, List[str]]
            Statistic names.
        vars : Optional[Union[str, List[str]]], optional
            Variables. If omitted, the statistics are removed for any variables. Default None.
        """
        path = self.get_filename(targs)
        with open(path, "r") as f:
            d = json.load(f)  # results is a list of dicts

        if isinstance(stats, str):
            stats = [stats]

        if vars is not None:
            raise NotImplementedError("TODO")

        for stat in stats:
            print(f"Removing {stat} in {os.path.basename(path)}")
            for item1 in d:
                for item2 in item1["stats"]:
                    item3 = item2["stats"]
                    for key in [f"{stat}", f"{stat}-time", f"{stat}-boot"]:
                        item3.pop(key, None)

        with open(path + ".tmp", "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False, indent=4)
        os.rename(path + ".tmp", path)

    def store(
        self,
        x_names: Union[str, List[str], Iterable[List[str]]],
        y_names: Union[str, List[str]],
        inputs_dict: Dict[str, Any],
        overwrite: bool = False,
        iterable_x: bool = False,
        save_every: int = 1,
        progress_bar: bool = True,
        total_iter: int = None,
        raise_error: bool = True,
    ) -> None:
        """
        Inputs_dict:
        - TODO
        """

        # Checks that the file is in the expected format.
        inputs_dict = self._check_inputs(inputs_dict)

        if isinstance(y_names, str):
            y_names = [y_names]

        y_names = sorted(y_names)

        # Create directory of not exists
        self.create(y_names)

        # Load existing data
        path = self.get_filename(y_names)
        with open(path, "r") as f:
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
            disable=not progress_bar,
        )
        is_changed = False
        for it, _x_names in enumerate(pbar, 1):
            if isinstance(_x_names, str):
                _x_names = [_x_names]

            _lines = list(set(_x_names))
            pbar.set_postfix({"x": str(_lines).replace("'", "")})

            _x_names = list(set(_x_names))
            _x_names = sorted(_x_names)

            # We check if the combination of variables already exists
            index_x = self._index_of(results, value=_x_names, key="x_names")

            # We create an entry if it doesn't exists
            if index_x is None:
                results.append(
                    {
                        "x_names": _x_names,
                        "stats": [],
                    }
                )
                index_x = -1

            # Ranges loop
            for restr in inputs_dict[
                "restrictions"
            ]:  # TODO: "restrictions" field can be None

                # We check if ranges already exists
                index_ranges = self._index_of_ranges(
                    results[index_x]["stats"], value=restr, key="restriction"
                )

                # We create an entry if it doesn't exists
                if index_ranges is None:
                    results[index_x]["stats"].append(
                        {
                            "restriction": restr,
                            "stats": {},
                        }
                    )
                    index_ranges = -1

                # Ranges restriction
                if self.restrictions is not None and restr is not None:
                    restrict_dict = self.restrictions[restr]
                elif restr is not None:
                    raise ValueError(
                        f"self.restriction must not be None when the restriction asked is not None (here {restr}). Consider using set_restrictions to load the dictionnary."
                    )
                else:
                    restrict_dict = {}

                _X, _Y = self.getter(
                    _x_names, y_names, restrict_dict, inputs_dict.get("max_samples")
                )

                entry = results[index_x]["stats"][index_ranges]["stats"]
                for stat in inputs_dict["statistics"]:
                    if overwrite and stat in entry:
                        entry.pop(stat, None)
                if set(entry.keys()) <= {"samples"}:  # If no keys or only "samples"
                    entry.pop("samples", None)

                for stat in inputs_dict["statistics"]:

                    if stat in entry:
                        continue
                    else:
                        is_changed = True
                    entry.update({stat: {}})

                    operator = self.stats[stat]

                    self._compute_stat(
                        _X,
                        _Y,
                        operator,
                        stat,
                        inputs_dict,
                        entry[stat],
                        raise_error=raise_error,
                    )  # Modify `entry` in-place

                samples = _Y.shape[0]
                prev_samples = entry.get("samples")
                if prev_samples is not None and samples != samples:
                    raise ValueError("Old and new number of samples are differents.")

                entry.update({"samples": samples})

            # Save results

            if it % save_every == 0 and is_changed:
                path = self.get_filename(y_names)
                with open(path + ".tmp", "w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=4)
                os.rename(path + ".tmp", path)
                is_changed = False

        # Final save

        if is_changed:
            path = self.get_filename(y_names)
            with open(path + ".tmp", "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=4)
            os.rename(path + ".tmp", path)

    def _compute_stat(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        operator: Statistic,
        stat: str,
        inputs_dict: Dict[str, Any],
        entry: Dict[str, Any],
        raise_error: bool = True,
    ) -> Dict[str, Any]:
        """
        TODO
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
            return {"value": None, "time": None, "std": None}

        # Simple computation
        try:
            start = time()
            value = operator(X, Y)
            end = time()
            entry.update(
                {
                    "value": value,
                    "time": end - start,
                }
            )
        except Exception as e:
            if raise_error:
                raise e
            entry.update({"value": None, "time": None, "std": None})
            return entry

        # Uncertainty
        if stat not in inputs_dict["uncertainty"]:
            return entry

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
        entry.update({"std": std})

        return entry

    # Reading access

    def read(
        self,
        x_names: Union[str, List[str], Iterable[List[str]]],
        y_names: Union[str, List[str]],
        restr: str,
        iterable_x: bool = False,
        default: str = "raise",
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Returns entries for variables `x_names` and targets `y_names`.

        Parameters
        ----------
        x_names : Union[str, List[str], Iterable[List[str]]]
            Variables. If Iterable, you must set the `iterable_x` argument to True.
        y_names : Union[str, List[str]]
            Targets.
        restr : str
            Restriction.
        iterable_x : bool, optional
            If True, the `x_names` argument is considered as a list of different variables. Default False.
        default : Any, optional
            Default behavior if entry does not exists. If "raise", an error is raised. Else, `default` is returned instead. By default "raise".

        Returns
        -------
        Union[Dict[str, Any], List[Dict[str, Any]]]
            Dictionnary corresponding to variables, targets and restrictions. If `iterable_x` is True, list of Dictionnary.
        """
        if isinstance(x_names, str):
            x_names = [x_names]

        if isinstance(y_names, str):
            y_names = [y_names]
        assert isinstance(y_names, List)

        # Load data
        path = self.get_filename(y_names)
        if not os.path.exists(path):
            raise FileNotFoundError(f"File {path} not exists yet.")
        with open(path, "r") as f:
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
                    raise ValueError(
                        f"Restriction of data {restr} doesn't exist in data"
                    )
                item = default

            # Store value
            item_list.append(item)

        if iterable_x:
            return item_list
        return item_list[0]

    def get_available_targets(self) -> List[List[str]]:
        """
        Returns all available targets in saves.

        Returns
        -------
        List[List[str]]
            Available targets in saves.
        """
        return [self.parse_filename(f) for f in self.get_existing_saves()]

    def get_available_variables(
        self,
        targets: Union[str, List[str]],
    ) -> List[List[str]]:
        """
        Returns all available variables for targets `y_names` in saves.

        Parameters
        ----------
        y_names : Union[None, str, List[str]]
            Targets.

        Returns
        -------
        List[List[str]]
            Available variables in saves.
        """
        # Load data
        path = self.get_filename(targets)
        if not os.path.exists(path):
            raise FileNotFoundError(f"File {path} not exists yet.")
        with open(path, "r") as f:
            data = json.load(f)  # results is a list of dicts

        return [item["x_names"] for item in data]

    def get_available_restrictions(
        self, targets: Union[str, List[str]], variables: Union[str, List[str]]
    ) -> List[str]:
        """
        Returns all available restrictions for targets `y_names` and variables `x_names` in saves.

        Parameters
        ----------
        x_names : Union[str, List[str]]
            Variables.
        y_names : Union[str, List[str]]
            Targets.

        Returns
        -------
        List[str]
            Available restrictions.
        """
        if isinstance(targets, str):
            targets = [targets]
        if isinstance(variables, str):
            variables = [variables]

        # Load data
        path = self.get_filename(targets)
        if not os.path.exists(path):
            raise FileNotFoundError(f"File {path} not exists yet.")
        with open(path, "r") as f:
            data = json.load(f)  # results is a list of dicts

        # Get variables content
        content = self._get_variables_content(variables, data)

        return [item["restriction"] for item in content]

    def get_available_stats(
        self,
        targets: Union[str, List[str]],
        variables: Union[str, List[str]],
        restriction: str,
    ) -> List[str]:
        """
        Returns all available statistics for targets `y_names`, variables `x_names` and restriction `restriction` in saves.

        Parameters
        ----------
        x_names : Union[str, List[str]]
            Variables.
        y_names : Union[str, List[str]]
            Targets.
        restriction : Union[str, List[str]]
            Restriction.

        Returns
        -------
        List[str]
            Available statistics.
        """
        if isinstance(targets, str):
            targets = [targets]
        if isinstance(variables, str):
            variables = [variables]

        # Load data
        path = self.get_filename(targets)
        if not os.path.exists(path):
            raise FileNotFoundError(f"File {path} not exists yet.")
        with open(path, "r") as f:
            data = json.load(f)  # results is a list of dicts

        # Get variables content
        content = self._get_variables_content(variables, data)
        if content is None:
            raise ValueError(f"Invalid variables {variables} for targets {targets}")

        for entry in content:
            if entry["restriction"] == restriction:
                return [s for s in entry["stats"].keys() if s != "samples"]

        raise ValueError(
            f"Invalid restriction {restriction} for targets {targets} and variables {variables}"
        )

    # Display

    def __str__(self):
        return "DiscreteHandler"

    # Helpers

    @staticmethod
    def _get_variables_content(
        x_names: List[str], data: List[Dict[str, Any]]
    ) -> Optional[List[Dict]]:
        """
        Returns the entry in `data` that match variable names `x_names`.
        If no entry matches, returns None.
        
        Parameters
        ----------
        x_names : List[str]
            Variable names.
        data : List[Dict[str, Any]]
            List of entries.

        Returns
        -------
        Optional[List[Dict]]
            Matching entry, if exists. Else None.
        """
        _x_names = set(x_names)
        for _item in data:
            if set(_item["x_names"]) == _x_names:
                item = _item["stats"]
                return item
        return None

    @staticmethod
    def _get_restriction_content(
        restr: str, data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Returns the entry in `data` that match restriction `restr`.
        If no entry matches, returns None.

        Parameters
        ----------
        restr : str
            Restriction name.
        data : List[Dict[str, Any]]
            List of entries.

        Returns
        -------
        Dict[str, Any]
            Matching entry, if exists. Else None.
        """
        for _item in data:
            if _item["restriction"] == restr:
                item = _item["stats"]
                return item
        return None

    @staticmethod
    def _index_of(
        ls: Sequence[Dict[str, Sequence]], value: Sequence, key: str
    ) -> Optional[int]:
        """
        Returns the index `i` of list `ls` with `ls[i][key] == value` where `value` is a list. Order of elements and case are important.

        Parameters
        ----------
        ls : Sequence[Dict[str, Sequence]]
            List of dictionnaries.
        value : Sequence
            Sequence to find.
        key : str
            Key of dictionnary to access `value`.

        Returns
        -------
        Optional[int]
            Index, if exists. Else None.
        """
        try:
            index = [tuple(item[key]) for item in ls].index(tuple(value))
        except ValueError:
            index = None
        return index

    @staticmethod
    def _index_of_ranges(
        ls: Sequence[Dict[str, Dict]], value: Sequence, key: str
    ) -> Optional[int]:
        """
        Returns the index `i` of list `ls` with `ls[i][key] == value` where `value` is a list. Order of elements and case are important.

        Parameters
        ----------
        ls : Sequence[Dict[str, Dict]]
            List of dictionnaries.
        value : Sequence
            Sequence to find.
        key : str
            Key of dictionnary to access `value`.

        Returns
        -------
        Optional[int]
            _description_
        """
        try:
            index = [item[key] for item in ls].index(value)
        except ValueError:
            index = None
        return index

    @staticmethod
    def _check_inputs(
        inputs_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Verifies the validity of `inputs_dict`. If necessary, can return a modified version of it. It does not modify the argument in-place.

        Parameters
        ----------
        inputs_dict : Dict[str, Any]
            Settings dictionnary for statistics computation.

        Returns
        -------
        Dict[str, Any]
            Potentially amended settings dictionnary.
        """
        inputs_dict = inputs_dict.copy()
        if "uncertainty" not in inputs_dict:
            inputs_dict.update({"uncertainty": {}})
        return inputs_dict
