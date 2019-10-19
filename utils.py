import json
import torch
from pathlib import Path
from typing import Union


class Config:
    """Config class"""

    def __init__(self, json_path_or_dict: Union[str, dict]) -> None:
        """Instantiating Config class

        Args:
            json_path_or_dict (Union[str, dict]): filepath of config or dictionary which has attributes
        """
        if isinstance(json_path_or_dict, dict):
            self.__dict__.update(json_path_or_dict)
        else:
            with open(json_path_or_dict, mode="r") as io:
                params = json.loads(io.read())
            self.__dict__.update(params)

    def save(self, json_path: str) -> None:
        """Saving config to json_path

        Args:
            json_path (str): filepath of config
        """
        with open(json_path, mode="w") as io:
            json.dump(self.__dict__, io, indent=4)

    def update(self, json_path_or_dict) -> None:
        """Updating Config instance

        Args:
            json_path_or_dict (Union[str, dict]): filepath of config or dictionary which has attributes
        """
        if isinstance(json_path_or_dict, dict):
            self.__dict__.update(json_path_or_dict)
        else:
            with open(json_path_or_dict, mode="r") as io:
                params = json.loads(io.read())
            self.__dict__.update(params)

    @property
    def dict(self):
        return self.__dict__


class CheckpointManager:
    def __init__(self, model_dir):
        if not isinstance(model_dir, Path):
            model_dir = Path(model_dir)

        if not model_dir.exists():
            model_dir.mkdir(parents=True)

        self._model_dir = model_dir

    def save_checkpoint(self, state, filename):
        torch.save(state, self._model_dir / filename)

    def load_checkpoint(self, filename):
        state = torch.load(self._model_dir / filename)
        return state


class SummaryManager:
    def __init__(self, model_dir):
        if not isinstance(model_dir, Path):
            model_dir = Path(model_dir)
        if not model_dir.exists():
            model_dir.mkdir(parents=True)

        self._model_dir = model_dir
        self._summary = {}

    def save(self, filename):
        with open(self._model_dir / filename, mode="w") as io:
            json.dump(self._summary, io, indent=4)

    def load(self, filename):
        with open(self._model_dir / filename, mode="r") as io:
            metric = json.loads(io.read())
        self.update(metric)

    def update(self, summary):
        self._summary.update(summary)

    def reset(self):
        self._summary = {}

    @property
    def summary(self):
        return self._summary
