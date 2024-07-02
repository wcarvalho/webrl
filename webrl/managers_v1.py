
from flask import session
from flask import render_template
from flask import redirect, url_for
from flask_socketio import emit
import random
from typing import Union, List

from webrl import types
import jax
from webrl import jax_utils
from webrl.types import ExperimentState
#class Config:
#    SECRET_KEY: str
#    DEBUG: int
#    PORT: int
    # Add other configuration variables here


class StageManager:

    def __init__(self,
                 index_file: str = 'index.html',
                 debug: bool = False):
        self.idx = 0
        self.state = None
        self.debug = debug
        self.index_called = False
        self.experiment_called = False
        self.index_file = index_file

    def set_state(self, state: ExperimentState):
        self.state = state

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

    def render_stage(self):
        return render_template(
            self.index_file,
            template_file=self.stage.html)

    def total_stages(self):
        return len(self.state.stages)

    def shift_stage(self, direction: str):
        stage_idx = self.state.stage_idx
        if direction == 'left':
            stage_idx -= 1
            stage_idx = max(0, stage_idx)
        elif direction == 'right':
            stage_idx += 1
            stage_idx = min(stage_idx, self.total_stages-1)
        else:
            raise NotImplementedError(direction)

        self.state = self.state.replace(stage_idx=stage_idx)
        self.stage.start_stage(self, self.stage)

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
        print("direction:", direction)
        self.shift_stage(direction)

    def split_rng(self):
        rng = jax_utils.unserialize_rng(session['rng'])
        rng, rng_ = jax.random.split(rng)
        session['rng'] = jax.serialize_rng(rng)
        self.update_stage(rng=rng)
        return rng_

    def upload_data(self):
        import ipdb; ipdb.set_trace()


class SessionManager:

    def __init__(self,
                 index_file: str = 'index.html',
                 debug: bool = False):
        self.idx = 0
        self.state = None
        self.debug = debug
        self.index_called = False
        self.experiment_called = False
        self.index_file = index_file

    def initialize(
        self,
        load_user: bool = True,
        initial_template: str = 'consent.html',
        experiment_fn_name: str = 'experiment',
    ):
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
            session.clear()
            SessionManager.reset_user()

        print(f"Rendering: {initial_template}")
        return render_template(self.index_file, template_file=initial_template)

    def init_state(self, stages, rng=None):
        return ExperimentState(
            unique_id=session['unique_id'],
            stages=stages,
            stage_idx=0,
            rng=rng,
        )
    def set_state(self, state: ExperimentState):
        self.state = state

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

    def render_stage(self):
        return render_template(
            self.index_file,
            template_file=self.stage.html)

    def total_stages(self):
        return len(self.state.stages)

    def shift_stage(self, direction: str):
        stage_idx = self.state.stage_idx
        if direction == 'left':
            stage_idx -= 1
            stage_idx = max(0, stage_idx)
        elif direction == 'right':
            stage_idx += 1
            stage_idx = min(stage_idx, self.total_stages-1)
        else:
            raise NotImplementedError(direction)

        self.state = self.state.replace(stage_idx=stage_idx)
        self.stage.start_stage(self, self.stage)

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
        print("direction:", direction)
        self.shift_stage(direction)

    def split_rng(self):
        rng = jax_utils.unserialize_rng(session['rng'])
        rng, rng_ = jax.random.split(rng)
        session['rng'] = jax.serialize_rng(rng)
        self.update_stage(rng=rng)
        return rng_

    def save_state(key, default=None):
        value = session.get(key, default)
        session[key] = value
        return value

    def upload_data(self):
        import ipdb; ipdb.set_trace()

    @staticmethod
    def reset_user():
        unique_id = random.getrandbits(32)
        user_seed = int(unique_id)
        session['unique_id'] = unique_id
        session['user_seed'] = user_seed

    @staticmethod
    def set(key, value):
        session[key] = value

    @staticmethod
    def get(key, default=None):
        value = session.get(key, default)
        session[key] = value
        return value

    @staticmethod
    def clear():
        session.clear()


