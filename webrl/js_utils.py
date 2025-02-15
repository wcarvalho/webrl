from typing import Optional, List, Callable, Any
from flask import render_template
from flask_socketio import emit
from flax import struct
import jax

from webrl.jax_utils import JaxWebEnvironment
from webrl.base_managers import BaseSessionManager, BaseStageManager
from webrl.session_manager import SessionManager


def update_html_content(
    stage_manager,
    return_only: bool = False,
    **kwargs):
  stage = stage_manager.stage
  content = {
      'title': stage_manager.make_stage_title(),
      'stage_idx': stage_manager.idx,
      'subtitle': stage.subtitle,
      'body': stage.body,
      # 'envcaption': stage.envcaption,
      **kwargs,
  }
  if return_only:
      return content
  emit('update_html_fields', content)

def update_environment_html_content(
      stage_manager,
      web_env,
      return_only: bool = False,
      **kwargs):
    stage = stage_manager.stage
    subtitle = stage.subtitle
    if stage.show_progress:
        subtitle += f"<br>Successes: {stage.num_success}/{stage.min_success}"
        subtitle += f"<br>Episodes: {stage.ep_idx}/{stage.max_episodes}"


    task = web_env.task_name(web_env.timestep) if stage.show_goal else ''
    content = {
        'title': stage_manager.make_stage_title(),
        'stage_idx': stage_manager.idx,
        'subtitle': subtitle,
        'taskDesc': task,
        'body': stage.body,
        'envcaption': stage.envcaption,
        **kwargs,
    }
    if return_only:
        print("return content only")
        return content
    emit('update_html_fields', content)

def start_stage(
        stage_manager,
        **kwargs):
    stage = stage_manager.stage
    emit('update_content', {
        'content': render_template(stage.html),
        'stage_idx': stage_manager.idx,
    })
    update_html_content(stage_manager=stage_manager)
    stage_manager.maybe_start_count_down()

def start_environment_stage(
        stage_manager,
        web_env,
        encode_locally=False,
        **kwargs):
    """New stage begins."""

    stage = stage_manager.stage
    if stage.restart:
        rng = SessionManager.split_rng(stage_manager)
        timestep = web_env.reset(stage.env_params, rng)
    else:
        timestep = web_env.timestep

    emit('update_content', {
        'content': render_template(stage.html),
        'stage_idx': stage_manager.idx,
    })

    raw_state, encoded_image = web_env.timestep_output(
        stage=stage,
        timestep=timestep,
        env_params=stage.env_params,
        encode_locally=encode_locally,
    )

    emit('action_taken', {
        'image': encoded_image,
        'state': raw_state,
    })

    stage.update_html_content(
        stage_manager=stage_manager,
        web_env=web_env,
    )
    stage_manager.maybe_start_count_down()

def start_upload_stage(
        stage_manager,
        **kwargs,
        ):
    stage = stage_manager.stage
    emit('update_content', {
        'content': render_template(stage.html),
        'stage_idx': stage_manager.idx,
    })
    update_html_content(
      stage_manager=stage_manager,
      stage=stage)

    stage_manager.upload_data()
    stage_manager.maybe_start_count_down()

def handle_stage_navigation_key_press(
        stage_manager,
        json,
        **kwargs):
    key = json['key']
    if key in ('ArrowLeft', 'ArrowRight'):
        direction = {
            'ArrowLeft': 'left',
            'ArrowRight': 'right',
        }[key]
        stage_manager.shift_stage(
            direction=direction)

def handle_environment_key_press(
        stage_manager,
        data_manager,
        web_env,
        json,
        encode_locally=False,
        **kwargs):

    key = json['key']

    stage = stage_manager.stage
    stage_idx = stage_manager.idx
    env_params = stage.env_params

    # Is the episode continuing? i.e. timestep from before is not last?
    if not web_env.timestep.last():
        if not web_env.keyparser.valid_key(key):
            return

        # update database with image, action, + times of each
        data_manager.update_in_episode_data(
            socket_json=json,
            stage_idx=stage_idx,
            web_env=web_env,
        )

        # take action
        rng = SessionManager.split_rng(stage_manager)
        timestep = web_env.step(key, env_params, rng)
        raw_state, encoded_image = web_env.timestep_output(
            stage=stage,
            timestep=timestep,
            env_params=stage.env_params,
            encode_locally=encode_locally,
        )

        emit('action_taken', {
            'image': encoded_image,
            'state': raw_state,
        })

        # is the NEXT time-step going to be last?
        if timestep.last():
            # evaluate success and update stage/info desc accordingly
            success = web_env.evaluate_success(timestep)
            stage_manager.update_stage(
                ep_idx=stage.ep_idx+1,
                num_success=stage.num_success + success,
                t=stage.t+1,
            )
            label = ''
            if stage.show_progress:
                label = 'SUCCESS' if success else 'FAILED'
                color = 'green' if success else 'red'
                label = f'<span style="color: {color}; font-weight: bold; font-size: 1.5em;">{label}!</span><br>'
            stage.update_html_content(
                stage_manager=stage_manager,
                web_env=web_env,
                taskDesc=f"{label}restarting. press any key to continue.",
            )
        else:
            # NOT last, just update time-step, nothing else
            stage_manager.update_stage(
                t=stage.t+1,
            )

    # episode is over
    else:

        # first episode data to master list
        data_manager.push_episode_data_to_master()

        ###################
        # check if this stage is over
        ###################
        ep_idx = stage.ep_idx
        num_success = stage.num_success

        max_episodes = stage.max_episodes
        min_success = stage.min_success

        achieved_min_success = num_success >= min_success
        achieved_max_episodes = ep_idx >= max_episodes

        go_to_next_stage = achieved_min_success or achieved_max_episodes
        # ------------
        # update to next stage
        # ------------
        if go_to_next_stage:
            data_manager.update_stage_data(
                stage=stage_manager.stage,
                stage_idx=stage_manager.idx,
            )

            # save all data
            SessionManager.save_progress(
                experiment_state=stage_manager.state,
                data_manager_state=dict(
                    all_episode_data=data_manager.all_episode_data,
                    stage_data=data_manager.stage_data),
                )
            stage_manager.shift_stage('right')

        # ------------
        # reset environment
        # ------------
        else:
            rng = SessionManager.split_rng(stage_manager)
            timestep = web_env.reset(stage.env_params, rng)
            raw_state, encoded_image = web_env.timestep_output(
                stage=stage,
                timestep=timestep,
                env_params=stage.env_params,
                encode_locally=encode_locally,
            )
            stage.update_html_content(
                stage_manager=stage_manager,
                web_env=web_env,
            )
            emit('action_taken', {
                'image': encoded_image,
                'state': raw_state,
            })

def advance_on_timer_finish(stage_manager, **kwargs):
    stage_manager.shift_stage('right')

def advance_no_interaction_timer(stage_manager, data_manager, web_env):
    ###################
    # store data and move to next stage
    ###################
    data_manager.update_in_episode_data(
        socket_json=dict(),
        stage_idx=stage_manager.idx,
        web_env=web_env,
        **{
            "key": 'none',
            "action": -1000,
        }
    )
    data_manager.update_stage_data(
        stage=stage_manager.stage,
        stage_idx=stage_manager.idx,
        **{
            't': 1,
            'ep_idx': 1,
            'num_success': 0,
        }
    )
    # save all data
    SessionManager.save_progress(
        experiment_state=stage_manager.state,
        data_manager_state=dict(episode_data=data_manager.all_episode_data,
                stage_data=data_manager.stage_data),
        )
    # Define the logic to be executed when the timer finishes
    stage_manager.shift_stage('right')


DefaultFnCall = Callable[
    [BaseStageManager, BaseSessionManager, JaxWebEnvironment], Any]

@struct.dataclass
class BaseStage:
    html: str
    title: Optional[str] = ''
    subtitle: Optional[str] = ''
    body: Optional[str] = ''

@struct.dataclass
class Stage(BaseStage):

    seconds: Optional[int] = None
    start_stage: DefaultFnCall = start_stage
    update_html_content: DefaultFnCall = update_html_content
    handle_key_press: DefaultFnCall = handle_stage_navigation_key_press
    handle_timer_finish: DefaultFnCall = advance_on_timer_finish



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

    start_stage: DefaultFnCall = start_environment_stage
    update_html_content: DefaultFnCall = update_environment_html_content
    handle_key_press: DefaultFnCall = handle_environment_key_press


