import itertools as itt
import os
import pickle
from copy import deepcopy
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from tqdm import tqdm

from ..stats.statistics import Statistic
from .handler import Handler

__all__ = ["ContinuousHandler"]


class ContinuousHandler(Handler):
    """
    Class for easily calculating, manipulating and saving calculations of statistical relationships between variables and targets estimated over sliding windows. The term continuous means that the calculation is performed for a large number of windows in order to approach a continuous result.
    """

    ext = ".pickle"   #: File extension
    filename_main_sep = "___"  #: Separator between targets and variables
    filename_secondary_sep = "+"  #: Separator between individual targets or variables

    def get_filename(
        self, x_names: Union[str, Sequence[str]], y_names: Union[str, Sequence[str]]
    ) -> str:
        """
        Builds a save filename from target names.

        Parameters
        ----------
        x_names : Union[str, Sequence[str]]
            Variable names.
        y_names : Union[str, Sequence[str]]
            Target names.

        Returns
        -------
        str
            Filename.
        """
        if isinstance(x_names, str):
            x_names = [x_names]
        if isinstance(y_names, str):
            y_names = [y_names]
        filename = self.filename_secondary_sep.join(sorted(y_names))\
            + self.filename_main_sep\
                + self.filename_secondary_sep.join(sorted(x_names))
        return os.path.join(self.save_path, filename + self.ext)

    def parse_filename(self, filename: str) -> Tuple[Sequence[str], Sequence[str]]:
        """
        Identifies variables and targets from formatted save filename.

        Parameters
        ----------
        filename : str
            Save filename.

        Returns
        -------
        Tuple[Sequence[str], Sequence[str]]
            Identified variables and targets.
        """
        assert filename.endswith(self.ext)
        filename = filename.removesuffix(self.ext)
        assert filename.count(self.filename_main_sep) == 1
        a, b = filename.split(self.filename_main_sep)
        return a.split(self.filename_secondary_sep), b.split(
            self.filename_secondary_sep
        )


    # Creation/removal

    def create(
        self, x_names: Union[str, Sequence[str]], y_names: Union[str, Sequence[str]]
    ) -> None:
        """
        Creates the statistics directory if not exists as well as the pickle files for features in `x_names` and `y_names`.

        Parameters
        ----------
        x_names : Union[str, Sequence[str]]
            Variable names.
        y_names : Union[str, Sequence[str]]
            Target names.
        """
        super().create()
        path = self.get_filename(x_names, y_names)
        if not os.path.isfile(path):
            with open(path, "wb") as f:
                pickle.dump([], f)

    def remove(
        self,
        variables: Optional[Sequence[Union[str, Sequence[str]]]],
        targets: Optional[Sequence[Union[str, Sequence[str]]]],
    ) -> None:
        """
        If `variables` and `targets` are both None, remove the whole directory if exists.
        If `variables` is None, remove all pickle files that match variables `targets`.
        If `targets` is None, remove all pickle files that match variables `variables`.
        If `variables` and `targets` are both not None, remove this specific pickle file.
        """
        if targets is None and variables is None:
            super().remove()
            return

        if variables is None:
            filenames = [
                self.get_filename(v, targets) for v in self.get_available_variables(targets)
            ]
        elif targets is None:
            filenames = [
                self.get_filename(variables, t) for t in self.get_available_targets()
            ]
        else:
            filenames = [
                self.get_filename(variables, targets)
            ]

        for name in filenames:
            if os.path.isfile(name):
                os.remove(name)

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
        if vars is None:
            vars = self.get_available_variables(targs)
        elif isinstance(vars, str):
            vars = [vars]

        for v in vars:
            path = self.get_filename(v, targs)

            with open(path, "rb") as f:
                d = pickle.load(f)

            if isinstance(stats, str):
                stats = [stats]

            for stat in stats:
                exists = False
                for item in d:
                    exists |= item["stats"].pop(stat, None) is not None

                if exists:
                    print(f"Removing {stat} in {os.path.basename(path)}")

            with open(path, "wb") as f:
                pickle.dump(d, f)

    # Writing access

    def store(
        self,
        x_names: Union[str, List[str]],
        y_names: Union[str, List[str]],
        inputs_dict: Dict[str, Any],
        overwrite: bool = False,
        raise_error: bool = True,
    ) -> None:
        """
        Inputs_dict:
        - TODO
        """
        if self.getter is None:
            raise RuntimeError(
                "You must call self.set_getter before calling self.update, self.overwrite or self.store."
            )

        # Checks that the file is in the expected format.
        inputs_dict = self._check_inputs(
            inputs_dict,
        )

        if isinstance(x_names, str):
            x_names = [x_names]
        if isinstance(y_names, str):
            y_names = [y_names]

        x_names, y_names = sorted(x_names), sorted(y_names)

        # Create directory of not exists
        self.create(x_names, y_names)

        # Load existing data
        path = self.get_filename(x_names, y_names)
        with open(path, "rb") as f:
            results = pickle.load(f)  # results is a list of dicts

        # Windows loop
        for wins in inputs_dict["windows"]:

            wins = wins.copy()
            # for key in ['features', 'bounds', 'scale', 'length', 'points']: TODO
            #     if key not in wins:
            #         wins[key] = [None]
            #     elif not isinstance(wins[key], List):
            #         wins[key] = [wins[key]]

            # We check if the data already exists
            index_windows = self._index_of_data(
                results, value=wins["features"], key="features"
            )

            # We create an entry if it doesn't exists
            if index_windows is None:
                results.append(
                    {
                        "features": wins["features"],
                        "stats": {},
                    }
                )
                index_windows = -1

            entry = results[index_windows]["stats"]
            for stat in inputs_dict["statistics"]:

                # We pass if the existing statistic exists and that overwrite is False
                if not overwrite and stat in entry:
                    continue
                if stat not in entry:
                    entry.update({stat: {}})

                operator = self.stats[stat]  # Callable(ndarray, ndarray) -> float

                self._compute_stat(
                    x_names,
                    y_names,
                    wins,
                    operator,
                    stat,
                    inputs_dict,
                    entry[stat],
                    raise_error=raise_error,
                )  # Modify `entry` in-place

                # Save results (update for each stat iteration)
                with open(self.get_filename(x_names, y_names), "wb") as f:
                    pickle.dump(results, f)

    # Reading access

    def read(
        self,
        x_names: Union[str, Sequence[str]],
        y_names: Union[str, Sequence[str]],
        wins_features: Union[str, Sequence[str]],
    ) -> Dict[str, Any]:
        # Formatting
        if isinstance(x_names, str):
            x_names = [x_names]
        if isinstance(y_names, str):
            y_names = [y_names]
        if isinstance(wins_features, str):
            wins_features = [wins_features]

        # Load data
        path = self.get_filename(x_names, y_names)
        if not os.path.exists(path):
            raise FileNotFoundError(f"File {path} not exists yet.")
        with open(path, "rb") as f:
            data = pickle.load(f)  # data is a list of dicts

        # Find the good windows
        names = set(wins_features)
        found = False
        for item in data:
            if set(item["features"]) == names:
                data = item
                found = True
                break
        if not found:
            raise ValueError(f"Windows {names} doesn't exist in data")

        return data

    def get_available_targets(self) -> List[List[str]]:
        """
        Returns all available targets in saves.

        Returns
        -------
        List[List[str]]
            Available targets in saves.
        """
        return self.drop_duplicates([self.parse_filename(f)[0] for f in self.get_existing_saves()])

    def get_available_variables(
        self, y_names: Union[None, str, List[str]]
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
        features = [self.parse_filename(f) for f in self.get_existing_saves()]
        if y_names is None:
            return [v for _, v in features]
        if isinstance(y_names, str):
            y_names = [y_names]
        return [v for t, v in features if set(t) == set(y_names)]

    def get_available_window_features(
        self, x_names: Union[str, List[str]], y_names: Union[str, List[str]]
    ) -> List[List[str]]:
        """
        Returns all available sliding window features for targets `y_names` and variables `x_names` in saves.

        Parameters
        ----------
        x_names : Union[str, List[str]]
            Variables.
        y_names : Union[str, List[str]]
            Targets.

        Returns
        -------
        List[List[str]]
            Available sliding window features.
        """
        filename = self.get_filename(x_names, y_names)
        if not os.path.isfile(filename):
            return []
        with open(filename, "rb") as f:
            data = pickle.load(f)
        return [item["features"] for item in data]

    def get_available_stats(
        self,
        x_names: Union[str, List[str]],
        y_names: Union[str, List[str]],
        window_features: Union[str, List[str]],
    ) -> List[str]:
        """
        Returns all available statistics for targets `y_names`, variables `x_names` and sliding window over `window_features` in saves.

        Parameters
        ----------
        x_names : Union[str, List[str]]
            Variables.
        y_names : Union[str, List[str]]
            Targets.
        window_features : Union[str, List[str]]
            Features for sliding window.

        Returns
        -------
        List[str]
            Available statistics.
        """
        filename = self.get_filename(x_names, y_names)
        if not os.path.isfile(filename):
            return []
        with open(filename, "rb") as f:
            data = pickle.load(f)
        for item in data:
            if set(item["features"]) == set(window_features):
                return list(item["stats"].keys())
        return []

    # Display

    def __str__(self):
        return "ContinuousHandler"

    # Helpers

    def _compute_stat(
        self,
        x_names: List[str],
        y_names: List[str],
        wins: Dict[str, Any],
        operator: Statistic,
        stat: str,
        inputs_dict: Dict[str, Any],
        entry: Dict[str, Any],
        raise_error: bool = True,  # TODO
    ) -> Dict[str, Any]:
        """
        wins:
        - features: List[str]
        - bounds: List[Tuple[float, float]]
        - bounds_include_windows: bool
        - scale: List[Literal[lin, log]]
        - length: List[float]
        - num_windows: List[float]
        - points: List[int]
        - overlap: List[Union[float, str]]

        Incompatible wins options:
        - length ou num_window
        - points ou overlap
        """

        if "length" not in wins:
            wins["length"] = [None] * len(wins["features"])
        if "points" not in wins:
            wins["points"] = [None] * len(wins["features"])

        coords = []
        bounds = []
        for i, ((low, upp), winsize, pts, sc) in enumerate(
            zip(wins["bounds"], wins["length"], wins["points"], wins["scale"])
        ):
            if winsize is None:
                winsize = (
                    (upp / low) ** (1 / wins["num_windows"][i])
                    if sc == "log"
                    else 1 / wins["num_windows"][i] * (upp - low)
                )

            if pts is None:
                ovp = wins["overlap"][i]
                if isinstance(ovp, str):
                    ovp = ovp.strip()
                    if ovp.endswith("%"):
                        ovp = float(ovp.removesuffix("%"))
                    else:
                        raise ValueError(
                            "Incorrect string overlap value (must include the % symbol)"
                        )
                    ovp = winsize / ovp

            if sc == "log":
                padd = np.sqrt(winsize) if wins["bounds_include_windows"] else 1.0
                if pts is None:
                    pts = (upp / low * padd**2) / ovp
                    pts = round(pts)
                xticks = np.logspace(np.log10(low * padd), np.log10(upp / padd), pts)
                bounds.append((xticks / padd, xticks * padd))
                coords.append(xticks)
            else:  # lin or None
                padd = winsize / 2 if wins["bounds_include_windows"] else 0.0
                if pts is None:
                    pts = (upp - low + 2 * padd) / ovp
                    pts = round(pts)
                xticks = np.linspace(low + padd, upp - padd, pts)
                bounds.append((xticks - padd, xticks + padd))
                coords.append(xticks)

            wins["points"] = [ticks.size for ticks in coords]

        data = np.zeros(wins["points"])
        samples = np.zeros(wins["points"])

        if (
            stat in inputs_dict["uncertainty"]
            and "name" in inputs_dict["uncertainty"][stat]
        ):
            std = np.zeros(wins["points"])
            name_resampling = inputs_dict["uncertainty"][stat]["name"]
            args = inputs_dict["uncertainty"][stat]["args"]
        else:
            std = None

        pbar = tqdm(
            itt.product(*[range(n) for n in wins["points"]]),
            total=np.prod(wins["points"]),
            leave=False,
            desc=f"Stat: {stat}, window: " + str(wins["features"]).replace("'", ""),
        )
        for ii in pbar:
            restrict = {
                key: [b[0][i], b[1][i]]
                for i, b, key in zip(ii, bounds, wins["features"])
            }
            _X, _Y = self.getter(x_names, y_names, restrict, inputs_dict["max_samples"])

            # Samples
            n = _Y.shape[0]
            samples[ii] = n

            # Computation of value
            if n >= inputs_dict["min_samples"]:
                try:
                    data[ii] = operator(_Y, _X)
                except:
                    data[ii] = np.nan
            else:
                data[ii] = np.nan

            if std is None:
                continue

            # Computation of estimator variance
            if n >= inputs_dict["min_samples"]:
                try:
                    std[ii] = self.resamplings[name_resampling].compute_sigma(
                        _X, _Y, operator, **args
                    )
                except:
                    std[ii] = np.nan
            else:
                std[ii] = np.nan

        entry.update(
            {"coords": tuple(coords), "data": data, "std": std, "samples": samples}
        )
        return entry

    @staticmethod
    def _index_of_data(
        ls: Sequence[Dict[str, Sequence]], value: Sequence, key: str
    ) -> Optional[int]:
        """
        Order of element and case are important.
        """
        try:
            index = [tuple(item[key]) for item in ls].index(tuple(value))
        except ValueError:
            index = None
        return index

    def _check_inputs(self, inputs_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        windows:
        - features: str or List[str]
          bounds: List[int, int] or List[List[int, int]]
          bounds_include_windows: bool or List[bool] (optional)
          scale: Literal[linear, log] or List[Literal[linear, log]] (optional)
          length: float or str or List[float or str] (optional, length ^ num_windows)
          num_windows: float or List[float] (optional, length ^ num_windows)
          points: int or List[int] (optional, points ^ overlap)
          overlap: float or str or List[float or str] (optional, points ^ overlap)

        min_samples: int (optional)
        max_samples: int (optional)

        statistics: List[str]

        uncertainty (optional):
            str:
                name: str
                args:
                    arg1: Any
                    ...
            ...
        """
        inputs_dict = deepcopy(inputs_dict)

        mandatory_keys = [
            "windows",
            "statistics",
        ]
        for key in mandatory_keys:
            assert key in inputs_dict

        optional_keys = ["min_samples", "max_samples", "uncertainty"]

        for key in inputs_dict:
            if key not in mandatory_keys:
                assert key in optional_keys

        key = "windows"
        if isinstance(inputs_dict[key], Dict):
            inputs_dict[key] = [inputs_dict[key]]
        assert isinstance(inputs_dict[key], List)
        assert all([isinstance(el, Dict) for el in inputs_dict[key]])
        mandatory_window_keys = ["features", "bounds"]
        optional_window_keys = ["bounds_include_windows", "scale"]
        special_window_keys_1 = ["length", "num_windows"]
        special_window_keys_2 = ["points", "overlap"]
        for d in inputs_dict[key]:
            assert set(d.keys()) <= set(
                mandatory_window_keys
                + optional_window_keys
                + special_window_keys_1
                + special_window_keys_2
            )
            for name in mandatory_window_keys:
                assert name in d
            assert sum([name in d for name in special_window_keys_1]) == 1
            assert sum([name in d for name in special_window_keys_2]) == 1

            if isinstance(d["features"], str):
                d["features"] = [d["features"]]
            n_features = len(d["features"])

            assert isinstance(d["features"], List)
            if isinstance(d["bounds"], List) and not isinstance(d["bounds"][0], List):
                d["bounds"] = [d["bounds"]] * n_features
            assert isinstance(d["bounds"], List)
            if "bounds_include_windows" in d:
                if isinstance(d["bounds_include_windows"], bool):
                    d["bounds_include_windows"] = [d["bounds_include_windows"]]
                assert isinstance(d["bounds_include_windows"], List)
            if "scale" in d:
                if isinstance(d["scale"], str):
                    d["scale"] = [d["scale"]] * n_features
                assert isinstance(d["scale"], List)
            if "length" in d:
                if isinstance(d["length"], (float, int)):
                    d["length"] = [d["length"]]
                assert isinstance(d["length"], List)
                assert len(d["length"]) == n_features
            if "num_windows" in d:
                if isinstance(d["num_windows"], (float, int)):
                    d["num_windows"] = [d["num_windows"]] * n_features
                assert isinstance(d["num_windows"], List)
            if "points" in d:
                if isinstance(d["points"], int):
                    d["points"] = [d["points"]] * n_features
                assert isinstance(d["points"], List)
            if "overlap" in d:
                if isinstance(d["overlap"], (str, float, int)):
                    d["overlap"] = [d["overlap"]] * n_features
                assert isinstance(d["overlap"], List)

            # Percents
            for name in ["length", "overlap"]:
                if name not in d:
                    continue
                for i, el in enumerate(d[name]):
                    if isinstance(el, str):
                        el = el.strip()
                        assert el[-1] == "%"
                        d[i] = el.strip()
                    assert isinstance(el, (float, int, str))

        key = "min_samples"
        if key not in inputs_dict:
            inputs_dict[key] = 0
        assert isinstance(inputs_dict[key], int)

        key = "max_samples"
        if key not in inputs_dict:
            inputs_dict[key] = None
        assert isinstance(inputs_dict[key], int) or inputs_dict[key] is None

        key = "statistics"
        if isinstance(inputs_dict[key], str):
            inputs_dict[key] = [inputs_dict[key]]
        assert isinstance(inputs_dict[key], List)
        assert all([isinstance(el, str) for el in inputs_dict[key]])

        key = "uncertainty"
        if key not in inputs_dict:
            inputs_dict[key] = {}
        assert isinstance(inputs_dict[key], Dict)
        assert all(
            [
                isinstance(k, str) and isinstance(d, Dict)
                for k, d in inputs_dict[key].items()
            ]
        )
        assert all(["name" in d for d in inputs_dict[key].values()])
        for k in inputs_dict[key]:
            assert inputs_dict[key][k]["name"] in self.resamplings
            if "args" not in inputs_dict[key][k]:
                inputs_dict[key][k] = {}

        return inputs_dict
