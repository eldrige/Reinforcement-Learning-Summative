"""Community Health Outreach environment.

Mission: a Community Health Worker (CHW) covers a rural district to run
non-communicable-disease screenings (blood pressure / blood glucose) in
households, refer high-risk residents to the health centre, give health
education to low-risk ones, and restock test kits at the clinic.

The district is a 10x10 grid split by a river that can only be crossed on
two bridges. Household positions are fixed (the district map); each
household's risk status is hidden and re-sampled every episode, so the agent
only learns it after screening.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

GRID = 10
CLINIC = (1, 1)
HOUSEHOLDS = [(2, 4), (3, 8), (0, 6), (7, 1), (8, 4), (6, 6), (9, 8), (7, 9)]
RIVER_X = 5
BRIDGES = (2, 7)  # y-coordinates where the river can be crossed
MAX_KITS = 5
MAX_STEPS = 200
HIGH_RISK_P = 0.4

# Action ids (exhaustive list of what a CHW does on a field day)
NORTH, SOUTH, WEST, EAST, SCREEN, REFER, RESTOCK, EDUCATE = range(8)
ACTION_NAMES = [
    "move_north", "move_south", "move_west", "move_east",
    "screen_household", "refer_to_clinic", "restock_kits", "health_education",
]
MOVES = {NORTH: (0, -1), SOUTH: (0, 1), WEST: (-1, 0), EAST: (1, 0)}

# Reward table
R_STEP = -0.02
R_BLOCKED = -0.3
R_INVALID = -0.5
R_SCREEN = 2.0
R_HIGH_RISK_FOUND = 1.0
R_REFER = 5.0
R_EDUCATE = 1.0
R_RESTOCK_EMPTY = 0.5  # restocking when out of kits is a good decision
R_SUCCESS = 10.0
R_TIME_BONUS = 5.0
R_MISSED_HIGH_RISK = -2.0
R_UNSCREENED = -1.0
SHAPING = 0.1  # potential-based guidance toward the nearest pending task

N_HH = len(HOUSEHOLDS)
HH_FEATS = 6  # dx, dy, screened, known_high_risk, referred, educated
OBS_DIM = 6 + 4 + 3 + N_HH * HH_FEATS  # status, route hint, task-here flags, households


def is_river(x: int, y: int) -> bool:
    return x == RIVER_X and y not in BRIDGES


def _bfs(src):
    """Walking distance (around the river) from src to every cell."""
    from collections import deque

    d = np.full((GRID, GRID), np.inf)
    d[src] = 0
    q = deque([src])
    while q:
        x, y = q.popleft()
        for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < GRID and 0 <= ny < GRID and not is_river(nx, ny) and d[nx, ny] == np.inf:
                d[nx, ny] = d[x, y] + 1
                q.append((nx, ny))
    return d


DIST_HH = [_bfs(h) for h in HOUSEHOLDS]
DIST_CLINIC = _bfs(CLINIC)


class CommunityHealthEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 6}

    def __init__(self, render_mode: str | None = None, renderer: str = "opengl", random_start: bool = False):
        super().__init__()
        self.random_start = random_start  # generalisation test: start anywhere in the district
        self.action_space = spaces.Discrete(len(ACTION_NAMES))
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(OBS_DIM,), dtype=np.float32)
        self.render_mode = render_mode
        self.renderer_kind = renderer
        self._renderer = None
        self.last_action = None
        self.last_reward = 0.0
        self.last_event = ""

    # ------------------------------------------------------------------ core
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.pos = list(CLINIC)  # start state: CHW leaves the health centre
        if self.random_start:
            while True:
                cand = [int(v) for v in self.np_random.integers(0, GRID, size=2)]
                if not is_river(*cand):
                    self.pos = cand
                    break
        self.kits = MAX_KITS
        self.t = 0
        high = self.np_random.random(N_HH) < HIGH_RISK_P
        if not high.any():  # every district day has at least one high-risk case
            high[self.np_random.integers(N_HH)] = True
        self.high_risk = high
        self.screened = np.zeros(N_HH, dtype=bool)
        self.referred = np.zeros(N_HH, dtype=bool)
        self.educated = np.zeros(N_HH, dtype=bool)
        self.episode_return = 0.0
        self.last_action, self.last_reward, self.last_event = None, 0.0, "Start at clinic"
        if self.render_mode == "human":
            self.render()
        return self._obs(), self._info()

    def _household_here(self) -> int | None:
        p = tuple(self.pos)
        return HOUSEHOLDS.index(p) if p in HOUSEHOLDS else None

    def _pending_dist_maps(self):
        maps = [DIST_HH[i] for i in range(N_HH)
                if (not self.screened[i] and self.kits > 0)
                or (self.screened[i] and self.high_risk[i] and not self.referred[i])]
        if self.kits == 0 and not self.screened.all():
            maps.append(DIST_CLINIC)
        return maps

    def _route_hint(self) -> list[float]:
        """One-hot of the move that walks toward the nearest outstanding task (the CHW's
        phone map route). All zeros when standing on a task or nothing is left."""
        maps = self._pending_dist_maps()
        x, y = self.pos
        hint = [0.0] * 4
        if not maps:
            return hint
        here = min(m[x, y] for m in maps)
        if here == 0:
            return hint
        best, best_a = here, None
        for a, (dx, dy) in MOVES.items():
            nx, ny = x + dx, y + dy
            if 0 <= nx < GRID and 0 <= ny < GRID and not is_river(nx, ny):
                d = min(m[nx, ny] for m in maps)
                if d < best:
                    best, best_a = d, a
        if best_a is not None:
            hint[best_a] = 1.0
        return hint

    def _obs(self) -> np.ndarray:
        x, y = self.pos
        hh = self._household_here()
        o = [x / (GRID - 1), y / (GRID - 1), self.kits / MAX_KITS,
             1.0 - self.t / MAX_STEPS, float(tuple(self.pos) == CLINIC),
             float(hh is not None)]
        o += self._route_hint()
        o += [float(hh is not None and not self.screened[hh]),
              float(hh is not None and self.screened[hh] and self.high_risk[hh] and not self.referred[hh]),
              float(hh is not None and self.screened[hh] and not self.high_risk[hh] and not self.educated[hh])]
        for i, (hx, hy) in enumerate(HOUSEHOLDS):
            known_high = self.screened[i] and self.high_risk[i]
            o += [(hx - x) / (GRID - 1), (hy - y) / (GRID - 1), float(self.screened[i]),
                  float(known_high), float(self.referred[i]), float(self.educated[i])]
        return np.asarray(o, dtype=np.float32)

    def _potential(self) -> float:
        """Negative walking distance to the nearest outstanding task."""
        x, y = self.pos
        dists = [m[x, y] for m in self._pending_dist_maps()]
        return -SHAPING * min(dists) if dists else 0.0

    def _mission_complete(self) -> bool:
        return bool(self.screened.all() and (self.referred | ~self.high_risk).all())

    def _info(self) -> dict:
        return {
            "screened": int(self.screened.sum()),
            "high_risk_found": int((self.screened & self.high_risk).sum()),
            "referred": int(self.referred.sum()),
            "high_risk_total": int(self.high_risk.sum()),
            "educated": int(self.educated.sum()),
            "kits": self.kits,
            "success": self._mission_complete(),
        }

    def step(self, action):
        action = int(action)
        self.t += 1
        r = R_STEP
        event = ACTION_NAMES[action]
        hh = self._household_here()
        phi_before = self._potential()

        if action in MOVES:
            dx, dy = MOVES[action]
            nx, ny = self.pos[0] + dx, self.pos[1] + dy
            if 0 <= nx < GRID and 0 <= ny < GRID and not is_river(nx, ny):
                self.pos = [nx, ny]
            else:
                r += R_BLOCKED
                event = "blocked (edge/river)"
        elif action == SCREEN:
            if hh is not None and not self.screened[hh] and self.kits > 0:
                self.kits -= 1
                self.screened[hh] = True
                r += R_SCREEN
                event = f"screened H{hh}"
                if self.high_risk[hh]:
                    r += R_HIGH_RISK_FOUND
                    event += " -> HIGH RISK"
            else:
                r += R_INVALID
                event = "invalid screen"
        elif action == REFER:
            if hh is not None and self.screened[hh] and self.high_risk[hh] and not self.referred[hh]:
                self.referred[hh] = True
                r += R_REFER
                event = f"referred H{hh} to clinic"
            else:
                r += R_INVALID
                event = "invalid referral"
        elif action == RESTOCK:
            if tuple(self.pos) == CLINIC and self.kits < MAX_KITS:
                if self.kits == 0:
                    r += R_RESTOCK_EMPTY
                self.kits = MAX_KITS
                event = "restocked kits"
            else:
                r += R_INVALID
                event = "invalid restock"
        elif action == EDUCATE:
            if hh is not None and self.screened[hh] and not self.high_risk[hh] and not self.educated[hh]:
                self.educated[hh] = True
                r += R_EDUCATE
                event = f"health talk H{hh}"
            else:
                r += R_INVALID
                event = "invalid education"

        r += self._potential() - phi_before
        terminated = self._mission_complete()
        truncated = False
        if terminated:
            r += R_SUCCESS + R_TIME_BONUS * (1.0 - self.t / MAX_STEPS)
            event += " | MISSION COMPLETE"
        elif self.t >= MAX_STEPS:
            truncated = True
            missed = int((self.high_risk & ~self.referred).sum())
            r += R_MISSED_HIGH_RISK * missed + R_UNSCREENED * int((~self.screened).sum())
            event += " | day over"

        self.episode_return += r
        self.last_action, self.last_reward, self.last_event = action, r, event
        if self.render_mode == "human":
            self.render()
        return self._obs(), float(r), terminated, truncated, self._info()

    # ------------------------------------------------------------- rendering
    def render(self):
        if self.render_mode is None:
            return None
        if self._renderer is None:
            from environment.rendering import make_renderer
            self._renderer = make_renderer(self.renderer_kind, self.render_mode)
        return self._renderer.draw(self)

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


if "CommunityHealth-v0" not in gym.registry:
    gym.register(id="CommunityHealth-v0", entry_point="environment.custom_env:CommunityHealthEnv")
