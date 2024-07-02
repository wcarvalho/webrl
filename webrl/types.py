from typing import Optional, List, Callable, Any
from flax import struct
import jax
import js_utils

from managers import StageManager, SessionManager
from jax_utils import JaxWebEnvironment

@struct.dataclass
class BaseStage:
    html: str
    title: Optional[str] = ''
    subtitle: Optional[str] = ''
    body: Optional[str] = ''

DefaultFnCall = Callable[
    [StageManager, SessionManager, JaxWebEnvironment], Any]

@struct.dataclass
class Stage(BaseStage):

    start_stage: DefaultFnCall = js_utils.start_stage
    update_html_content: DefaultFnCall = js_utils.update_html_content
    handle_key_press: DefaultFnCall = js_utils.handle_stage_navigation_key_press
    handle_timer_finish: js_utils.advance_on_timer_finish



@struct.dataclass
class EnvironmentStage(Stage):
    envcaption: Optional[str] = ''
    env_params: Optional[struct.PyTreeNode] = None
    restart: bool = True
    show_progress: bool = True
    show_goal: bool = True
    max_episodes: Optional[int] = 10
    min_success: Optional[int] = 1
    count_down_started: bool = False
    t: int = 0
    ep_idx: int = 0
    num_success: int = 0

    start_stage: DefaultFnCall = js_utils.start_environment_stage
    update_html_content: DefaultFnCall = js_utils.update_environment_html_content
    handle_key_press: DefaultFnCall = js_utils.handle_environmention_key_press


@struct.dataclass
class ExperimentState:
    unique_id: str
    stages: List[Stage]
    stage_idx: int
    rng: jax.Array
