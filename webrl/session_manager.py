
import sys
import pickle
import importlib
from flask import session
from flask import render_template
from flask import redirect, url_for
from flax import serialization
from flax import struct
from flax.traverse_util import flatten_dict, unflatten_dict
from safetensors.flax import save_file, load_file

import random
from typing import List, Callable
import os.path

import jax
from webrl import jax_utils
from webrl import base_managers as bases


@struct.dataclass
class ExperimentState:
    unique_id: str
    stages: List[struct.PyTreeNode]
    stage_idx: int
    rng: jax.numpy.ndarray


def save_pytree(pytree: struct.PyTreeNode, filename: str):

    def remove_callable(x):
        if isinstance(x, Callable):
            return None
        return x
    pytree = jax.tree_map(remove_callable, pytree)

    # Get the class of the pytree
    cls = type(pytree)

    # Get the module and class name
    module_name = cls.__module__
    class_name = cls.__name__

    # Serialize the pytree
    serialized_state = serialization.to_bytes(pytree)

    # Create a dictionary with all the necessary information
    data = {
        'module_name': module_name,
        'class_name': class_name,
        'state': serialized_state
    }


    # Ensure the directory exists
    directory = os.path.dirname(filename)
    if directory:
        os.makedirs(directory, exist_ok=True)

    # Save everything using pickle
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def load_pytree(filename: str) -> struct.PyTreeNode:
    with open(filename, 'rb') as f:
        data = pickle.load(f)

    # Extract the information
    module_name = data['module_name']
    class_name = data['class_name']
    serialized_state = data['state']

    # Import the module and get the class
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)

    # Deserialize the state
    return serialization.from_bytes(cls, serialized_state)

class SessionManager(bases.BaseSessionManager):

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

    @staticmethod
    def split_rng(stage_manager):
        rng = jax_utils.unserialize_rng(session['rng'])
        rng, rng_ = jax.random.split(rng)
        session['rng'] = jax_utils.serialize_rng(rng)
        stage_manager.set_state(stage_manager.state.replace(rng=rng))
        return rng_

    @staticmethod
    def save_progress(experiment_state: ExperimentState,
                      data_manager_state: struct.PyTreeNode):
        user_seed = session['user_seed']

        # save experiment state
        filename = f'data/{user_seed}_experiment_state.safetensors'
        save_pytree(experiment_state, filename)
        print(f"saved: {filename}")

        # save data manager state
        filename = f'data/{user_seed}_data_manager_state.safetensors'
        save_pytree(data_manager_state, filename)
        print(f"saved: {filename}")

    @staticmethod
    def load_progress():
        user_seed = session['user_seed']

        experiment_state = None
        data_manager_state = None
        # load experiment state
        filename = f'data/{user_seed}_experiment_state.safetensors'
        if os.path.exists(filename):
            experiment_state = load_pytree(filename)
            print(f"loaded: {filename}")


        # load experiment state
        if os.path.exists(filename):
            filename = f'data/{user_seed}_data_manager_state.safetensors'
            data_manager_state = load_pytree(filename)
            print(f"loaded: {filename}")

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

