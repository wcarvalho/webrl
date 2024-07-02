
from flask import session
from flask import render_template
from flask import redirect, url_for
from flask_socketio import emit
from flax import serialization
from flax import struct
from flax.traverse_util import flatten_dict, unflatten_dict
from safetensors.flax import save_file, load_file

import random
from typing import Union, List, Optional, Callable

from webrl import types
import jax
from webrl import jax_utils
from webrl.types import ExperimentState
from jax_utils import JaxWebEnvironment


class SessionManager:

    def __init__(self, index_file: str = 'index.html', debug: bool = False):
        self.idx = 0
        self.state = None
        self.debug = debug
        self.index_file = index_file
        self.index_called = False
        self.experiment_called = False

    def initialize(
        self,
        load_user: bool = True,
        initial_template: str = 'consent.html',
        experiment_fn_name: str = 'experiment',
    ):
        """This is the ONLY file where the session manager manages HTML content.

        This only happens when the session begins."""
        session.permanent = load_user
        self.index_called = True
        if load_user:
            if 'unique_id' not in session:
                print("Loading new user")
                SessionManager.reset_user()
            else:
                print(f"loading prior user: {session['unique_id']}")
                print(f"Redirecting to: {experiment_fn_name}")
                return redirect(url_for(experiment_fn_name))
        else:
            print("Resetting user")
            SessionManager.clear()
            SessionManager.reset_user()

        return render_template(self.index_file, template_file=initial_template)


    def split_rng(self, stage_manager):
        rng = jax_utils.unserialize_rng(session['rng'])
        rng, rng_ = jax.random.split(rng)
        session['rng'] = jax.serialize_rng(rng)
        stage_manager.set_state(stage_manager.state.replace(rng=rng))
        return rng_

    def save_progress(experiment_state: ExperimentState,
                      data_manager_state: struct.PyTreeNode):
        user_seed = session['user_seed']

        # save experiment state
        filename = f'{user_seed}_experiment_state.safetensors'
        flattened_dict = flatten_dict(experiment_state, sep=',')
        save_file(flattened_dict, filename)

        # save data manager state
        filename = f'{user_seed}_data_manager_state.safetensors'
        flattened_dict = flatten_dict(data_manager_state, sep=',')
        save_file(flattened_dict, filename)

    def load_progress():
        user_seed = session['user_seed']

        # load experiment state
        filename = f'{user_seed}_experiment_state.safetensors'
        flattened_dict = load_file(filename)
        experiment_state = unflatten_dict(flattened_dict, sep=',')


        # load experiment state
        filename = f'{user_seed}_data_manager_state.safetensors'
        flattened_dict = load_file(filename)
        data_manager_state = unflatten_dict(flattened_dict, sep=',')
        return experiment_state, data_manager_state

    @staticmethod
    def reset_user():
        unique_id = random.getrandbits(32)
        user_seed = int(unique_id)
        session['unique_id'] = unique_id
        session['user_seed'] = user_seed

    @staticmethod
    def __setitem__(key, value):
        session[key] = value

    @staticmethod
    def set(key, value):
        session[key] = value

    @staticmethod
    def __getitem__(key):
        value = session[key]
        return value

    @staticmethod
    def get(key, default=None, overwrite: bool = True):
        value = session.get(key, default)
        if overwrite:
            session[key] = value
        return value

    @staticmethod
    def clear():
        session.clear()



class DataManager:

    def __init__(self):
        self.episode_data = []
        self.all_episode_data = []
        self.stage_data = []

    def update_stage_data(
            self,
            stage,
            stage_idx,
            **kwargs):
        def remove_callable(x):
            if isinstance(x, Callable): return None
            return x
        stage = jax.tree_map(remove_callable, stage)
        stage = jax_utils.serialize(stage)

        # Convert stage to a state dict
        new_row = serialization.to_state_dict(stage)

        # Create the new_row dictionary
        new_row.update({
            "stage_idx": stage_idx,
            'unique_id': int(SessionManager['user_seed']),
            **kwargs
        })
        import ipdb; ipdb.set_trace()

        self.stage_data.append(new_row)

    def push_episode_data_to_master(self):
        """add current episode to list containing all episodes"""
        self.all_episode_data.extend(self.episode_data)


    def update_in_episode_data(
            self,
            socket_json: dict,
            stage_idx: int,
            web_env,
            **kwargs):
        """update database with image, action, + times of each."""

        timestep = web_env.timestep
        success = web_env.evaluate_success(timestep)

        timestep = timestep.replace(observation=None)
        timestep = jax_utils.serialize(timestep)

        action = web_env.keyparser.action(socket_json.get('key', None))
        action = int(action) if action else action


        new_row = {
            "stage_idx": int(stage_idx),
            "image_seen_time": str(socket_json['imageSeenTime']),
            "key_press_time": str(socket_json['keydownTime']),
            "key": str(socket_json['key']),
            "success": success,
            "action": action,
            "timestep": timestep,
            "rng": list(jax_utils.serialize_rng(SessionManager['rng'])),
            'unique_id': int(SessionManager['user_seed']),
            **kwargs
        }
        import ipdb; ipdb.set_trace()

        self.episode_data.append(new_row)



class StageManager:

    def __init__(self,
                 index_file: str = 'index.html',
                 debug: bool = False):
        self.idx = 0
        self.state = None
        self.debug = debug
        self.index_file = index_file

    def init_state(self, session_manager, stages, rng=None):
        return ExperimentState(
            unique_id=session_manager['unique_id'],
            stages=stages,
            stage_idx=0,
            rng=rng,
        )

    def set_state(self, state: ExperimentState):
        self.state = state

    def render_stage(self):
        return self.render_template(self.index_file, template_file=self.stage.html)

    @property
    def idx(self):
        if self.state is None:
            import ipdb; ipdb.set_trace()
        return self.state.stage_idx

    @property
    def stage(self):
        if self.state is None:
            import ipdb; ipdb.set_trace()
        return self.state.stages[self.state.stage_idx]

    def update_stage(self, **kwargs):
        stage = self.stage.replace(**kwargs)
        self.state.stages[self.state.stage_idx] = stage

    def make_stage_title(self, stage):
        if self.debug:
            return f"{self.idx}/{self.total_stages}. {self.stage.title}"
        else:
            return self.stage.title

    def update_html_content(self):
        self.stage.update_html_content()

    def render_stage(self):
        return render_template(
            self.index_file,
            template_file=self.stage.html)

    def total_stages(self):
        return len(self.state.stages)

    def decrement_stage_idx(self):
        stage_idx = self.state.stage_idx
        stage_idx -= 1
        stage_idx = max(0, stage_idx)
        self.state = self.state.replace(stage_idx=stage_idx)

    def incremenet_stage_idx(self):
        stage_idx = self.state.stage_idx
        stage_idx += 1
        stage_idx = min(stage_idx, self.total_stages-1)
        self.state = self.state.replace(stage_idx=stage_idx)

    def shift_stage(self, direction: str):
        if direction == 'left':
            self.decrement_stage_idx()
        elif direction == 'right':
            self.incremenet_stage_idx()
        else:
            raise NotImplementedError(direction)

        self.stage.start_stage(self)

    def maybe_start_count_down(self):
        seconds = self.stage.seconds
        if seconds:
            count_down_started = self.stage.count_down_started
            if not count_down_started:
                print('starting timer: start_env_stage')
                emit('start_timer', {'seconds': seconds})
            self.update_stage(count_down_started=True)

    def record_click(self,
                     json: dict,
                     web_env: Optional[JaxWebEnvironment]=None):
        direction = json['direction']
        self.shift_stage(direction)

    def handle_key_press(self,
                        json: dict,
                        web_env: Optional[JaxWebEnvironment]=None,
                        data_manager: Optional[DataManager]=None,
                        ):
        """This happens INSIDE a stage"""
        self.stage.handle_key_press(
            stage_manager=self,
            data_manager=data_manager,
            web_env=web_env,
            json=json)

    def handle_timer_finish(
            self,
            web_env: Optional[JaxWebEnvironment]=None,
            data_manager: Optional[DataManager]=None,
            ):
        emit('stop_timer')
        self.stage.handle_timer_finish(
            stage_manager=self,
            web_env=web_env,
            data_manager=data_manager,
        )
