
from base64 import b64encode
import jax
import jax.numpy as jnp
import json
from flask_socketio import emit
from flax import serialization
import io
from PIL import Image
import numpy as np


def serialize_rng(rng: jax.Array):
    return tuple(int(i) for i in rng)

def unserialize_rng(rng_tuple):
    return jnp.array(rng_tuple, dtype=jnp.uint32)

def convert_to_serializable(obj):
    """Convert JAX objects to Python objects"""
    if isinstance(obj, (jnp.ndarray, np.ndarray)):
        if isinstance(obj, jnp.ndarray):
            obj = jax.tree_map(np.asarray, obj)
        return obj.tolist()  # Convert JAX array to list
    #elif isinstance(obj, (int, float, str, bool)):
    #    return obj
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(v) for v in obj]
    #elif hasattr(obj, '__dict__'):
    #    return array_to_python(vars(obj))
    else:
        return obj

def serialize_pytree(pytree, jsonify: bool = True):
    pytree = serialization.to_state_dict(pytree)
    pytree = convert_to_serializable(pytree)
    if jsonify:
        return json.dumps(pytree)
    return pytree


def encode_image(state_image):
    buffer = io.BytesIO()

    Image.fromarray(state_image.astype('uint8')).save(buffer, format="JPEG")
    encoded_image = b64encode(buffer.getvalue()).decode('ascii')
    return 'data:image/jpeg;base64,' + encoded_image

def default_timestep_output(
        stage,
        timestep,
        encode_locally: bool = False,
        **kwargs):
    if encode_locally:
         state_image = stage.render_fn(timestep, stage.env_params)
         processed_image = encode_image(state_image)
         return None, processed_image
    else:
        state = serialize_pytree(timestep.state, jsonify=False)
        return state, None

def default_evaluate_success(timestep):
    return int(timestep.reward > .8)

class JaxWebEnvironment:

    def __init__(
            self,
            env,
            keyparser,
            timestep_output_fn = None,
            evaluate_success_fn = None,
            task_name_fn = None
            ):
        self.env = env
        self.keyparser = keyparser
        self.timestep = None

        if timestep_output_fn is None:
            timestep_output_fn = default_timestep_output
        self.timestep_output = timestep_output_fn
        
        if evaluate_success_fn is None:
            evaluate_success_fn = default_evaluate_success
        self.evaluate_success = evaluate_success_fn

        if task_name_fn is None:
            def task_name_fn(*args, **kwargs): 'task'
        self.task_name = task_name_fn


    def reset(self, env_params, rng):
        self.timestep = self.env.reset(rng, env_params)

        return self.timestep

    def step(self, action_key, env_params, rng):
        action = self.keyparser.action(action_key)
        self.timestep = self.env.step(
            rng, self.timestep, action, env_params)

        return self.timestep

