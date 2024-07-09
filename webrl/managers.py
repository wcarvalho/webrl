
from flask import render_template
from flask_socketio import emit
from flask import render_template_string
from flax import serialization
from flax import struct
import json

from typing import List, Optional, Callable, List
import os.path

import jax
from webrl import jax_utils
from webrl.jax_utils import JaxWebEnvironment
from webrl import base_managers as bases
from webrl.session_manager import SessionManager, ExperimentState

class DataManager(bases.BaseDataManager):

    def __init__(self):
        self.in_episode_data = []
        self.load_state(self.init_state())

    def init_state(self):
        return dict(
            all_episode_data=[],
            stage_data=[]
        )

    def load_state(self, state: struct.PyTreeNode):
        all_episode_data = state.get('all_episode_data')
        if all_episode_data is not None:
            self.all_episode_data = all_episode_data
            print("loaded all_episode_data")
        stage_data = state.get('stage_data')
        if stage_data is not None:
            self.stage_data = stage_data
            print("loaded stage_data")


    def update_stage_data(
            self,
            stage,
            stage_idx,
            **kwargs):
        def remove_callable(x):
            if isinstance(x, Callable): return None
            return x
        stage = jax.tree_map(remove_callable, stage)
        stage = jax_utils.serialize_pytree(stage)

        # Convert stage to a state dict
        new_row = serialization.to_state_dict(stage)

        # Create the new_row dictionary
        new_row.update({
            "stage_idx": stage_idx,
            'unique_id': int(SessionManager.get('user_seed')),
            **kwargs
        })
        import ipdb; ipdb.set_trace()

        self.stage_data.append(new_row)

    def push_episode_data_to_master(self):
        """add current episode to list containing all episodes"""
        self.all_episode_data.extend(self.in_episode_data)

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
        timestep = jax_utils.serialize_pytree(timestep, jsonify=False)

        action = web_env.keyparser.action(socket_json.get('key', None))
        action = int(action) if action else action


        new_row = {
            "stage_idx": int(stage_idx),
            "image_seen_time": str(socket_json['imageSeenTime']),
            "key_press_time": str(socket_json['keydownTime']),
            "key": str(socket_json['key']),
            "success": success,
            "action": action,
            "timestep": json.dumps(timestep),
            "rng": list(jax_utils.serialize_rng(SessionManager.get('rng'))),
            'unique_id': int(SessionManager.get('user_seed')),
            # purely for convenience
            **{f"state_{k}": json.dumps(v) for k, v in timestep['state'].items()},
            **kwargs
        }

        self.in_episode_data.append(new_row)

class StageManager(bases.BaseStageManager):

    def __init__(
            self,
            app,
            stages: List[struct.PyTreeNode],
            data_manager: DataManager,
            web_env: JaxWebEnvironment,
            index_file: str = 'index.html',
            debug: bool = False):
        self.app = app
        self._state = None
        self.debug = debug
        self.stages = stages
        self.index_file = index_file
        self.data_manager = data_manager
        self.web_env = web_env

    def init_state(self, rng=None):
        if rng is None:
            rng = SessionManager.get('rng')
            if rng:
                rng = jax_utils.unserialize_rng(rng)
            else:
                rng = jax.random.PRNGKey(SessionManager.get('user_seed'))

        return ExperimentState(
            unique_id=SessionManager.get('unique_id'),
            stages=self.stages,
            stage_idx=0,
            rng=rng,
        )

    def set_state(self, state: ExperimentState):
        self._state = state
        SessionManager.set('rng', jax_utils.serialize_rng(state.rng))
        print(f"Set session rng to {state.rng}=={SessionManager.get('rng')}")

    def render_stage(self):
        return self.render_template(self.index_file, template_file=self.stage.html)

    def maybe_load_state(self):
        if self._state is None:
            print("-"*25)
            print('trying loading user_seed', SessionManager.get('user_seed'))

            experiment_state, data_manager_state = SessionManager.load_progress(
                example_experiment_state=self.init_state(),
                example_data_manager_state=self.data_manager.init_state(),
            )
            if not experiment_state and not data_manager_state:
                print("Nothing found. resetting state...")
                experiment_state = self.init_state()
                data_manager_state = self.data_manager.init_state()

            self.set_state(experiment_state)
            print(f'LOADED experiment_state: {experiment_state.summary()}')

            self.data_manager.load_state(data_manager_state)
            print('LOADED data_manager_state')
            return True
        return False
    @property
    def state(self):
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

    def make_stage_title(self):
        if self.debug:
            return f"{self.idx+1}/{self.total_stages}. {self.stage.title}"
        else:
            return self.stage.title


    def start_stage(self):
        self.stage.start_stage(
            stage_manager=self,
            web_env=self.web_env)

    def update_html_content(self):
        self.stage.update_html_content(
            stage_manager=self,
            web_env=self.web_env,
            )

    def render_stage(self):
        return render_template(
            self.index_file,
            template_file=self.stage.html)

    def render_stage_template(self):
       with self.app.app_context():
        index_file_path = os.path.join(
        self.app.template_folder, self.index_file)
        with open(index_file_path, 'r') as file:
            template_content = file.read()
        emit('update_content', {
            'template': template_content,
        })
        self.stage.start_stage(
            stage_manager=self,
            web_env=self.web_env,
        )

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

    def shift_stage(self, direction: str):
        if direction == 'left':
            self.decrement_stage_idx()
        elif direction == 'right':
            self.incremenet_stage_idx()
        else:
            raise NotImplementedError(direction)

        self.stage.start_stage(
            stage_manager=self,
            web_env=self.web_env)

    def maybe_start_count_down(self):
        seconds = self.stage.seconds
        if seconds:
            count_down_started = self.stage.count_down_started
            if not count_down_started:
                print('starting timer: start_env_stage')
                emit('start_timer', {'seconds': seconds})
            self.update_stage(count_down_started=True)

    def record_click(self, json: dict):
        direction = json['direction']
        self.shift_stage(direction)

    def handle_key_press(self, json: dict):
        """This happens INSIDE a stage"""
        self.stage.handle_key_press(
            stage_manager=self,
            data_manager=self.data_manager,
            web_env=self.web_env,
            json=json)

    def handle_timer_finish(self):
        emit('stop_timer')
        self.stage.handle_timer_finish(
            stage_manager=self,
            web_env=self.web_env,
            data_manager=self.data_manager,
        )
