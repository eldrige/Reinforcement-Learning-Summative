import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from environment.custom_env import (
    CLINIC, EDUCATE, HOUSEHOLDS, MAX_KITS, MAX_STEPS, OBS_DIM, REFER, RESTOCK, SCREEN,
    CommunityHealthEnv, is_river,
)


def test_gymnasium_api():
    check_env(CommunityHealthEnv(), skip_render_check=True)


def test_reset_start_state():
    env = CommunityHealthEnv()
    obs, info = env.reset(seed=0)
    assert obs.shape == (OBS_DIM,)
    assert tuple(env.pos) == CLINIC and env.kits == MAX_KITS
    assert env.high_risk.any() and info["screened"] == 0


def test_river_blocks_movement():
    env = CommunityHealthEnv()
    env.reset(seed=0)
    env.pos = [4, 5]
    assert is_river(5, 5)
    _, r, *_ = env.step(3)  # east into river
    assert env.pos == [4, 5] and r < 0


def test_screen_refer_and_invalid_actions():
    env = CommunityHealthEnv()
    env.reset(seed=0)
    env.high_risk[:] = False
    env.high_risk[0] = True
    env.pos = list(HOUSEHOLDS[0])
    _, r, *_ = env.step(SCREEN)
    assert env.screened[0] and env.kits == MAX_KITS - 1 and r > 2
    _, r, *_ = env.step(SCREEN)          # already screened -> invalid
    assert r < 0
    _, r, *_ = env.step(EDUCATE)         # high risk needs referral, not education
    assert r < 0
    _, r, *_ = env.step(REFER)
    assert env.referred[0] and r > 4
    _, r, *_ = env.step(RESTOCK)         # not at clinic
    assert r < 0


def test_mission_complete_terminates():
    env = CommunityHealthEnv()
    env.reset(seed=1)
    env.screened[:] = True
    env.referred[:] = env.high_risk
    env.screened[-1] = False
    env.referred[-1] = False
    env.high_risk[-1] = False
    env.pos = list(HOUSEHOLDS[-1])
    _, r, term, trunc, info = env.step(SCREEN)
    assert term and not trunc and info["success"] and r > 10


def test_truncation_at_time_limit():
    env = CommunityHealthEnv()
    env.reset(seed=2)
    done = False
    while not done:
        _, _, term, trunc, _ = env.step(RESTOCK)
        done = term or trunc
    assert trunc and env.t == MAX_STEPS


@pytest.mark.parametrize("seed", [0, 1])
def test_torchrl_wrapper(seed):
    from training.common import make_env

    env = make_env(seed=seed)
    td = env.rollout(10)
    assert td["observation"].shape == (10, OBS_DIM)
    assert np.isfinite(td["next", "reward"].numpy()).all()
