import json
import os
from time import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union
from types import NoneType

import numpy as np
from tqdm import tqdm

from ..stats.statistics import Statistic
from .handler import Handler

__all__ = ["DiscreteHandler"]


class DiscreteHandler(Handler):
    """
    Class for easily calculating, manipulating and saving calculations of statistical relationships between variables and targets according to predefined situations (restrictions). The term “discrete” means that the calculation is performed for a finite number of independent restrictions.
    """

    ext = ".json"  #: File extension
    restrictions: Optional[Dict[str, Dict]] = None  #: Dict of current restrictions
    filename_sep: str = "_"  #: Separator between targets

    # Setter

    def set_restrictions(
        self,
        d: Dict[str, Dict[str, Tuple[float, float]]]
    ) -> None:
        """
        Set new restrictions, i.e., the constraints on one or more targets that reduce the number of data samples that can be used in the calculation.

        Parameters
        ----------
        d : Dict[str, Dict[str, Tuple[float, float]]]
            New restrictions.
        """
        if not isinstance(d, Dict):
            raise TypeError(f"Restriction dictionnary must be a Dict, not {type(d)}")
        t = self._check_dict_type(d, str, (Dict, NoneType))
        if t is not None:
            raise TypeError(
                f"{t} is not a valid entry ({type(t[0])}, {type(t[1])} instead of (str, Dict))"
            )

        for _d in d.values():
            if _d is None:
                continue
            t = self._check_dict_type(_d, str, List)
            if t is not None:
                raise TypeError(
                    f"{t} is not a valid entry ({type(t[0])}, {type(t[1])} instead of (str, Dict))"
                )

        self.restrictions = d

    # Saves

    def get_filename(
        self,
        y_names: Union[str, Sequence[str]]
    ) -> str:
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

    def parse_filename(
        self,
        filename: str
    ) -> Sequence[str]:
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

    def create(
        self,
        y_names: Union[str, Sequence[str]]
    ):
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

    def remove(
        self,
        y_names: Optional[Union[List[str], str]]
    ) -> None:
        """
        Removes saved results.
        If `y_names` is None, remove the entire `self.save_path` directory.
        If `y_names` is not None, only remove the corresponding JSON file, if exists. If not, raise an error.

        Parameters
        ----------
        y_names : Optional[Union[List[str], str]]
            Name of target file to remove. If None, all saves are deleted.
        """
        if not os.path.isdir(self.save_path):
            raise FileNotFoundError(f"Save directory {self.save_path} does not exist.")

        if y_names is None:
            super().remove()
            return

        path = self.get_filename(y_names)
        if os.path.isfile(path):
            os.remove(path)

    def delete_stats(
        self,
        x_names: Optional[Union[str, List[str]]],
        y_names: Union[str, List[str]],
        stats: Union[str, List[str]],
    ) -> None:
        """
        Removes stats `stats` for variables `x_names` and targets `y_names` .
        If `x_names` is omitted, the stats are removed for any variable with the specified target.

        Parameters
        ----------
        x_names : Optional[Union[str, List[str]]], optional
            Variable names. If None, the statistics are removed for any variables.
        y_names : Union[str, List[str]]
            Target names.
        stats : Union[str, List[str]]
            Statistic names.
        """
        path = self.get_filename(y_names)
        with open(path, "r") as f:
            d = json.load(f)  # results is a list of dicts

        if isinstance(stats, str):
            stats = [stats]

        if isinstance(x_names, str):
            x_names = [x_names]

        for stat in stats:
            print(f"Removing {stat} in {os.path.basename(path)}")
            for item1 in d:
                if x_names is not None and set(item1["x_names"]) != set(x_names):
                    continue
                for item2 in item1["stats"]:
                    item3 = item2["stats"]
                    for key in [f"{stat}", f"{stat}-time", f"{stat}-boot"]:
                        item3.pop(key, None)

        with open(path + ".tmp", "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False, indent=4)
        os.rename(path + ".tmp", path)

    @staticmethod
    def check_settings(
        settings: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Verifies the validity of `settings`. If necessary, can return a modified version of it. It does not modify the dictionnary in-place.

        Parameters
        ----------
        settings : Dict[str, Any]
            Settings dictionnary for statistics computation. Format:

            - `statistics`: List[str] -- names of statistics to compute. If you want to use a custom statistic, consider calling `set_additional_stats` before.
            - `restrictions`: List[str], optional -- names of restrictions to use. The definition of the restriction are provided by the `set_restrictions` method.
            - `uncertainty`: Dict[key, entry], optional -- key is a statistic name and entry is a Dict with keys "name" (field to provide the name of the Resampler to use) and "args" (field to provide keyword arguments for Resampler). If you want to use a custom resampling, consider calling `set_additional_resamplings` before.
            - `min_samples`: int, optional -- minimum number of samples to use for computation. If the actual number of available samples is lower, the result is set to None.
            - `max_samples`: int, optional -- maximum number of samples to use for computation. If the actual number of available samples is higher, `max_samples` random samples are drawn.
            
        Returns
        -------
        Dict[str, Any]
            Potentially amended settings dictionnary.
        """
        settings = settings.copy()

        if "statistics" not in settings or settings["statistics"] is None:
            raise ValueError("'statistics' field in `settings` dictionnary must be a list of strings")
        
        if "uncertainty" not in settings or settings["uncertainty"] is None:
            settings.update({"uncertainty": {}})
        for stat in settings["uncertainty"]:
            if "name" not in settings["uncertainty"][stat]:
                raise ValueError("Uncertainty entries must provide a method name under the field 'name'.")
            if "args" not in settings["uncertainty"][stat]:
                settings["uncertainty"][stat].update({"args": {}})

        if "restriction" not in settings or settings["restriction"] is None:
            settings.update({"restriction": {}})
        if "min_samples" not in settings:
            settings.update({"min_samples": None})
        if "max_samples" not in settings:
            settings.update({"max_samples": None})
        return settings

    def store(
        self,
        x_names: Union[str, List[str], Iterable[List[str]]],
        y_names: Union[str, List[str]],
        settings: Dict[str, Any],
        overwrite: bool = False,
        iterable_x: bool = False,
        save_every: int = 1,
        progress_bar: bool = True,
        total_iter: int = None,
        raise_error: bool = True,
    ) -> None:
        """
        Computes and saves statistics. Detailed instructions are provided by `settings`. If `overwrite` is True, existing results are overwritten. Else, they are kept.
        If `iterable_x` is True, the function assumes that `x_names` is an list of variables or sets of variables.
        
        Parameters
        ----------
        x_names : Union[str, List[str], Iterable[List[str]]]
            Variable or set of variable names. If `iterable_x` is True, list of variable or set of variable names.
        y_names : Union[str, List[str]]
            Target or set of target names.
        settings : Dict[str, Any]
            Instructions for computation. More details on the dictionnary format are given in the `check_settings` documentation. 
        overwrite : bool, optional
            Whether existing results must be overwritten, by default False (existing results kept).
        iterable_x : bool, optional
            Whether `x_names` is a list of variables or sets of variables, by default False.
        save_every : int, optional
            Defines how many variables the backup should be updated with. Increasing the value of this argument speeds up the program by reducing the number of times the backup file is written (ignored if `iterable_x` is False), by default 1
        progress_bar : bool, optional
            Whether a progress bar has to be displayed (ignored if `iterable_x` is False), by default True.
        total_iter : int, optional
            Number of elements in iterable. Useful when the iterable is not a Sequence, by default None.
        raise_error : bool, optional
            Whether the function should propagate errors that occur during the calculation of statistics. If False, the entries are set to None, by default True.
        """

        # Checks that the file is in the expected format.
        settings = self.check_settings(settings)

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

        if not iterable_x:
            progress_bar = False

        if total_iter is None and iterable_x and isinstance(x_names, (List, Tuple)):
            total_iter = len(x_names)

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
            for restr in settings[
                "restrictions"
            ]:
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
                    _x_names, y_names, restrict_dict, settings.get("max_samples")
                )

                entry = results[index_x]["stats"][index_ranges]["stats"]
                for stat in settings["statistics"]:
                    if overwrite and stat in entry:
                        entry.pop(stat, None)
                if set(entry.keys()) <= {"samples"}:  # If no keys or only "samples"
                    entry.pop("samples", None)

                for stat in settings["statistics"]:

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
                        settings,
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
        settings: Dict[str, Any],
        entry: Dict[str, Any],
        raise_error: bool = True,
    ) -> Dict[str, Any]:
        """
        Modifies in-place the `entry` dictionnary.
        Adds or changes the three following entries:
        - `value`: computed only if current value is None.
            If error, the default value is None.
        - `time`: computed only if new value of `stat` is not None.
        - `std`: computed only if new value of `stat` is not None and if `settings["bootstrapping"][stat]` is not None.
        If the available number of samples is lower than `settings["min_samples"]`,
        then all three entries are set to None.

        Parameters
        ----------
        X : np.ndarray
            Variable data.
        Y : np.ndarray
            Target data.
        operator : Statistic
            Statistic estimator.
        stat : str
            Statistics name.
        settings : Dict[str, Any]
            Instructions for computation. More details on the dictionnary format are given in the `check_settings` documentation. 
        entry : Dict[str, Any]
            Statistics entry.
        raise_error : bool, optional
            Whether the function should propagate errors that occur during the calculation of statistics. If False, the entries are set to None, by default True.

        Returns
        -------
        Dict[str, Any]
            Updated entry.
        """
        # Samples
        samples = Y.shape[0]
        if samples <= settings["min_samples"]:
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
        if stat not in settings["uncertainty"]:
            return entry

        try:
            d = settings["uncertainty"][stat]
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
        y_names: Union[str, List[str]],
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
        path = self.get_filename(y_names)
        if not os.path.exists(path):
            raise FileNotFoundError(f"File {path} not exists yet.")
        with open(path, "r") as f:
            data = json.load(f)  # results is a list of dicts

        return [item["x_names"] for item in data]

    def get_available_restrictions(
        self,
        x_names: Union[str, List[str]],
        y_names: Union[str, List[str]]
    ) -> List[str]:
        """
        Returns all available restrictions for targets variables `x_names` and `y_names` in saves.

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
        if isinstance(x_names, str):
            x_names = [x_names]
        if isinstance(y_names, str):
            y_names = [y_names]

        # Load data
        path = self.get_filename(y_names)
        if not os.path.exists(path):
            raise FileNotFoundError(f"File {path} not exists yet.")
        with open(path, "r") as f:
            data = json.load(f)  # results is a list of dicts

        # Get variables content
        content = self._get_variables_content(x_names, data)

        return [item["restriction"] for item in content]

    def get_available_stats(
        self,
        x_names: Union[str, List[str]],
        y_names: Union[str, List[str]],
        restriction: str,
    ) -> List[str]:
        """
        Returns all available statistics for variables `x_names`, targets `y_names`, and restriction `restriction` in saves.

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
        if isinstance(x_names, str):
            x_names = [x_names]
        if isinstance(y_names, str):
            y_names = [y_names]

        # Load data
        path = self.get_filename(y_names)
        if not os.path.exists(path):
            raise FileNotFoundError(f"File {path} not exists yet.")
        with open(path, "r") as f:
            data = json.load(f)  # results is a list of dicts

        # Get variables content
        content = self._get_variables_content(x_names, data)
        if content is None:
            raise ValueError(f"Invalid variables {x_names} for targets {y_names}")

        for entry in content:
            if entry["restriction"] == restriction:
                return [s for s in entry["stats"].keys() if s != "samples"]

        raise ValueError(
            f"Invalid restriction {restriction} for variables {x_names} and targets {y_names}"
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
        restr: str,
        data: List[Dict[str, Any]]
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
        ls: Sequence[Dict[str, Sequence]],
        value: Sequence,
        key: str
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
        ls: Sequence[Dict[str, Dict]],
        value: Sequence, key: str
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
