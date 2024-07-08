import abc
from typing import Optional, Any, Dict

class BaseSessionManager(abc.ABC):
    @abc.abstractmethod
    def __init__(self, index_file: str, debug: bool):
        pass

    @abc.abstractmethod
    def initialize(self, load_user: bool, initial_template: str, experiment_fn_name: str):
        pass

    @abc.abstractmethod
    def split_rng(self, stage_manager):
        pass

    @staticmethod
    @abc.abstractmethod
    def save_progress(experiment_state, data_manager_state):
        pass

    @staticmethod
    @abc.abstractmethod
    def load_progress():
        pass

    @staticmethod
    @abc.abstractmethod
    def reset_user():
        pass

    @staticmethod
    @abc.abstractmethod
    def __setitem__(key, value):
        pass

    @staticmethod
    @abc.abstractmethod
    def __getitem__(key):
        pass

    @staticmethod
    @abc.abstractmethod
    def get(key, default=None, overwrite: bool = True):
        pass

    @staticmethod
    @abc.abstractmethod
    def clear():
        pass

class BaseDataManager(abc.ABC):
    @abc.abstractmethod
    def __init__(self):
        pass

    @abc.abstractmethod
    def update_stage_data(self, stage, stage_idx, **kwargs):
        pass

    @abc.abstractmethod
    def push_episode_data_to_master(self):
        pass

    @abc.abstractmethod
    def update_in_episode_data(self, socket_json: dict, stage_idx: int, web_env, **kwargs):
        pass

    @abc.abstractmethod
    def load_state(self, state):
        pass

class BaseStageManager(abc.ABC):
    @abc.abstractmethod
    def __init__(self, data_manager, index_file: str, debug: bool):
        pass

    @abc.abstractmethod
    def init_state(self, session_manager, stages, rng=None):
        pass

    @abc.abstractmethod
    def set_state(self, state):
        pass

    @abc.abstractmethod
    def render_stage(self):
        pass

    @property
    @abc.abstractmethod
    def state(self):
        pass

    @property
    @abc.abstractmethod
    def idx(self):
        pass

    @property
    @abc.abstractmethod
    def stage(self):
        pass

    @abc.abstractmethod
    def update_stage(self, **kwargs):
        pass

    @abc.abstractmethod
    def make_stage_title(self, stage):
        pass

    @abc.abstractmethod
    def update_html_content(self):
        pass

    @abc.abstractmethod
    def total_stages(self):
        pass

    @abc.abstractmethod
    def decrement_stage_idx(self):
        pass

    @abc.abstractmethod
    def incremenet_stage_idx(self):
        pass

    @abc.abstractmethod
    def shift_stage(self, direction: str):
        pass

    @abc.abstractmethod
    def maybe_start_count_down(self):
        pass

    @abc.abstractmethod
    def record_click(self, json: dict, web_env: Optional[Any] = None):
        pass

    @abc.abstractmethod
    def handle_key_press(self, json: dict, web_env: Optional[Any] = None):
        pass

    @abc.abstractmethod
    def handle_timer_finish(self, web_env: Optional[Any] = None):
        pass