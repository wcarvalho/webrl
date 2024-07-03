
from flask import render_template
from flask_socketio import emit
from flax import serialization
from flax import struct

from typing import List, Optional, Callable

import jax
from webrl import jax_utils
from webrl.jax_utils import JaxWebEnvironment
from webrl import base_managers as bases
from webrl.session_manager import SessionManager, ExperimentState

class DataManager(bases.BaseDataManager):

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

    def load_state(self, state: struct.PyTreeNode):
        self.all_episode_data = state['all_episode_data']
        self.stage_data = state['stage_data']


class StageManager(bases.BaseStageManager):

    def __init__(self,
                 data_manager: DataManager,
                 index_file: str = 'index.html',
                 debug: bool = False):
        self._state = None
        self.debug = debug
        self.index_file = index_file
        self.data_manager = data_manager

    def init_state(self, session_manager, stages, rng=None):
        return ExperimentState(
            unique_id=session_manager['unique_id'],
            stages=stages,
            stage_idx=0,
            rng=rng,
        )

    def set_state(self, state: ExperimentState):
        self._state = state

    def render_stage(self):
        return self.render_template(self.index_file, template_file=self.stage.html)

    @property
    def state(self):
        if self._state is None:
            experiment_state, data_manager_state = SessionManager.load_progress()
            import ipdb; ipdb.set_trace()
            self.set_state(experiment_state)
            self.data_manager.load_state(data_manager_state)

        return self._state

    @property
    def idx(self):
        return self.state.stage_idx

    @property
    def stage(self):
        return self.state.stages[self.state.stage_idx]

    @property
    def total_stages(self):
        return len(self.state.stages)

    def update_stage(self, **kwargs):
        stage = self.stage.replace(**kwargs)
        self.state.stages[self.state.stage_idx] = stage

    def make_stage_title(self, stage):
        if self.debug:
            return f"{self.idx}/{self.total_stages}. {self.stage.title}"
        else:
            return self.stage.title

    def update_html_content(self):
        self.stage.update_html_content(self)

    def render_stage(self):
        return render_template(
            self.index_file,
            template_file=self.stage.html)

    def decrement_stage_idx(self):
        stage_idx = self.state.stage_idx
        stage_idx -= 1
        stage_idx = max(0, stage_idx)
        self.set_state(self.state.replace(stage_idx=stage_idx))

    def incremenet_stage_idx(self):
        stage_idx = self.state.stage_idx
        stage_idx += 1
        stage_idx = min(stage_idx, self.total_stages-1)
        self.set_state(self.state.replace(stage_idx=stage_idx))

    def shift_stage(self, direction: str, web_env: JaxWebEnvironment):
        if direction == 'left':
            self.decrement_stage_idx()
        elif direction == 'right':
            self.incremenet_stage_idx()
        else:
            raise NotImplementedError(direction)

        self.stage.start_stage(
            stage_manager=self,
            web_env=web_env)

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
        self.shift_stage(direction, web_env=web_env)

    def handle_key_press(self,
                        json: dict,
                        web_env: Optional[JaxWebEnvironment]=None,
                        ):
        """This happens INSIDE a stage"""
        self.stage.handle_key_press(
            stage_manager=self,
            data_manager=self.data_manager,
            web_env=web_env,
            json=json)

    def handle_timer_finish(
            self,
            web_env: Optional[JaxWebEnvironment]=None,
            ):
        emit('stop_timer')
        self.stage.handle_timer_finish(
            stage_manager=self,
            web_env=web_env,
            data_manager=self.data_manager,
        )
