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

    def parse_filename(
        self,
        filename: str
    ) -> Tuple[Sequence[str], Sequence[str]]:
        """
        Identifies variables and targets from formatted save filename.

        Parameters
        ----------
        filename : str
            Save filename.

        Returns
        -------
        Sequence[str]
            Identified variables.
        Sequence[str]]
            Identified targets.
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
        self,
        x_names: Union[str, Sequence[str]],
        y_names: Union[str, Sequence[str]]
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
        x_names: Optional[Sequence[Union[str, Sequence[str]]]],
        y_names: Optional[Sequence[Union[str, Sequence[str]]]],
    ) -> None:
        """
        Removes saved results.
        If `x_names` and `y_names` are both None, remove the whole directory if exists.
        If `x_names` is None, remove all pickle files that match targets `y_names`.
        If `y_names` is None, remove all pickle files that match variables `x_names`.
        If `x_names` and `y_names` are both not None, remove this specific pickle file.

        Parameters
        ----------
        x_names : Optional[Sequence[Union[str, Sequence[str]]]]
            Variable names. If None, remove all files that match targets `y_names`.
        y_names : Optional[Sequence[Union[str, Sequence[str]]]]
            Target names. If None, remove all files that match variables `x_names`.
        """
        if y_names is None and x_names is None:
            super().remove()
            return

        if x_names is None:
            filenames = [
                self.get_filename(v, y_names) for v in self.get_available_variables(y_names)
            ]
        elif y_names is None:
            filenames = [
                self.get_filename(x_names, t) for t in self.get_available_targets()
            ]
        else:
            filenames = [
                self.get_filename(x_names, y_names)
            ]

        for name in filenames:
            if os.path.isfile(name):
                os.remove(name)

    def delete_stats(
        self,
        x_names: Optional[Union[str, List[str]]],
        y_names: Union[str, List[str]],
        stats: Union[str, List[str]],
    ) -> None:
        """
        Removes stats `stats` for targets `y_names` and variables `x_names`.
        If `x_names` is omitted, the stats are removed for any variable with the specified target.

        Parameters
        ----------
        x_names : Optional[Union[str, List[str]]]
            Variables. If None, the statistics are removed for any variables.
        y_names : Union[str, List[str]]
            Targets.
        stats : Union[str, List[str]]
            Statistic names.
        """
        if x_names is None:
            x_names = self.get_available_variables(y_names)
        elif isinstance(x_names, str):
            x_names = [x_names]

        for v in x_names:
            path = self.get_filename(v, y_names)

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

    def check_settings(
        self,
        settings: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Verifies the validity of `settings`. If necessary, can return a modified version of it. It does not modify the dictionnary in-place.

        Parameters
        ----------
        settings : Dict[str, Any]
            Settings dictionnary for statistics computation. Format:

            - `statistics`: List[str] -- names of statistics to compute. If you want to use a custom statistic, consider calling `set_additional_stats` before.
            - `windows`:

                * `features`: str or List[str] -- sliding window features. These may be targets other than those for which the statistics are calculated.
                * `bounds`: List[int, int] or List[List[int, int]] -- bounds [start, stop] of sliding windows. The order is the same as for the features list.
                * `bounds_include_windows`: bool or List[bool], optional -- whether the bounds correspond to the limits for the center of the sliding windows (False) or to the side of the windows (True).
                * `scale`: Literal[linear, log] | List[Literal[linear, log]], optional -- scale of sliding windows. Note that the scale can differ between two sliding windows.
                * `length`: float | str | List[float or str], optional -- length of the window. If scale is linear, it correspond to an additive offset. If scale is log, it correspond to a multiplicative factor between the two extremities of the window. One, and only one, field among `length` and `num_windows` has to be provided.
                * `num_windows`: float or List[float], optional -- number of non-overlapping sliding window for each feature.  One, and only one, field among `length` and `num_windows` has to be provided.
                * `points`: int or List[int], optional -- number of sliding windows. One, and only one, field among `points` and `overlap` has to be provided.
                * `overlap`: float or str or List[float or str], optional -- Percentage of overlap between two consecutive sliding windows (the interpretation depends on the chosen scale). This constrain the number of windows. One, and only one, field among `points` and `overlap` has to be provided.

            - `min_samples`: int, optional -- minimum number of samples to use for computation. If the actual number of available samples is lower, the result is set to NaN.
            - `max_samples`: int, optional -- maximum number of samples to use for computation. If the actual number of available samples is higher, `max_samples` random samples are drawn.
            - `uncertainty`: Dict[key, entry], optional -- key is a statistic name and entry is a Dict with keys "name" (field to provide the name of the Resampler to use) and "args" (field to provide keyword arguments for Resampler). If you want to use a custom resampling, consider calling `set_additional_resamplings` before.

        Returns
        -------
        Dict[str, Any]
            Potentially amended settings dictionnary.
        """
        settings = deepcopy(settings)

        mandatory_keys = [
            "windows",
            "statistics",
        ]
        for key in mandatory_keys:
            assert key in settings

        optional_keys = ["min_samples", "max_samples", "uncertainty"]

        for key in settings:
            if key not in mandatory_keys:
                assert key in optional_keys

        key = "windows"
        if isinstance(settings[key], Dict):
            settings[key] = [settings[key]]
        assert isinstance(settings[key], List)
        assert all([isinstance(el, Dict) for el in settings[key]])
        mandatory_window_keys = ["features", "bounds"]
        optional_window_keys = ["bounds_include_windows", "scale"]
        special_window_keys_1 = ["length", "num_windows"]
        special_window_keys_2 = ["points", "overlap"]
        for d in settings[key]:
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
        if key not in settings:
            settings[key] = 0
        assert isinstance(settings[key], int)

        key = "max_samples"
        if key not in settings:
            settings[key] = None
        assert isinstance(settings[key], int) or settings[key] is None

        key = "statistics"
        if isinstance(settings[key], str):
            settings[key] = [settings[key]]
        assert isinstance(settings[key], List)
        assert all([isinstance(el, str) for el in settings[key]])

        key = "uncertainty"
        if key not in settings:
            settings[key] = {}
        assert isinstance(settings[key], Dict)
        assert all(
            [
                isinstance(k, str) and isinstance(d, Dict)
                for k, d in settings[key].items()
            ]
        )
        assert all(["name" in d for d in settings[key].values()])
        for k in settings[key]:
            assert settings[key][k]["name"] in self.resamplings
            if "args" not in settings[key][k]:
                settings[key][k] = {}

        return settings

    def store(
        self,
        x_names: Union[str, List[str]],
        y_names: Union[str, List[str]],
        settings: Dict[str, Any],
        overwrite: bool = False,
        raise_error: bool = False,
    ) -> None:
        """
        Computes and saves statistics. Detailed instructions are provided by `settings`. If `overwrite` is True, existing results are overwritten. Else, they are kept.

        Parameters
        ----------
        x_names : Union[str, List[str]]
            Variable or set of variable names.
        y_names : Union[str, List[str]]
            Target or set of target names.
        settings : Dict[str, Any]
            Instructions for computation. More details on the dictionnary format are given in the `check_settings` documentation.
        overwrite : bool, optional
            Whether existing results must be overwritten, by default False (existing results kept).
        raise_error : bool, optional
            Whether the function should propagate errors that occur during the calculation of statistics. If False, the entries are set to None, by default True.
        """
        if self.getter is None:
            raise RuntimeError(
                "You must call self.set_getter before calling self.update, self.overwrite or self.store."
            )

        # Checks that the file is in the expected format.
        settings = self.check_settings(
            settings,
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
        for wins in settings["windows"]:

            wins = wins.copy()

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
            for stat in settings["statistics"]:

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
                    settings,
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
        """
        Returns entries for variables `x_names`, targets `y_names` and sliding window features `wins_features`.

        Parameters
        ----------
        x_names : Union[str, Sequence[str]]
            Variable names.
        y_names : Union[str, Sequence[str]]
            Target names.
        wins_features : Union[str, Sequence[str]]
            Sliding window feature names.

        Returns
        -------
        Dict[str, Any]
            _description_
        """
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

        return data["stats"]

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
        self,
        y_names: Union[None, str, List[str]]
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
        self,
        x_names: Union[str, List[str]],
        y_names: Union[str, List[str]]
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
        settings: Dict[str, Any],
        entry: Dict[str, Any],
        raise_error: bool = False,
    ) -> Dict[str, Any]:
        """
        _summary_

        Parameters
        ----------
        x_names : List[str]
            Variable names.
        y_names : List[str]
            Target names.
        wins : Dict[str, Any]
            Sliding window feature names.
        operator : Statistic
            Statistic estimator.
        stat : str
            Statistic name.
        settings : Dict[str, Any]
            Instructions for computation. More details on the dictionnary format are given in the `check_settings` documentation. 
        entry : Dict[str, Any]
            Statistics entry.
        raise_error : bool, optional
            Whether the function should propagate errors that occur during the calculation of statistics. If False, estimate are set to NaN, by default False.

        Returns
        -------
        Dict[str, Any]
            Updated entry.
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
            stat in settings["uncertainty"]
            and "name" in settings["uncertainty"][stat]
        ):
            std = np.zeros(wins["points"])
            name_resampling = settings["uncertainty"][stat]["name"]
            args = settings["uncertainty"][stat]["args"]
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
            _X, _Y = self.getter(x_names, y_names, restrict, settings["max_samples"])

            # Samples
            n = _Y.shape[0]
            samples[ii] = n

            # Computation of value
            if n >= settings["min_samples"]:
                try:
                    data[ii] = operator(_Y, _X)
                except Exception as e:
                    if raise_error is True:
                        raise e
                    data[ii] = np.nan
            else:
                data[ii] = np.nan

            if std is None:
                continue

            # Computation of estimator variance
            if n >= settings["min_samples"]:
                try:
                    std[ii] = self.resamplings[name_resampling].compute_sigma(
                        _X, _Y, operator, **args
                    )
                except Exception as e:
                    if raise_error is True:
                        raise e
                    std[ii] = np.nan
            else:
                std[ii] = np.nan

        entry.update(
            {"coords": tuple(coords), "data": data, "std": std, "samples": samples}
        )
        return entry

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
