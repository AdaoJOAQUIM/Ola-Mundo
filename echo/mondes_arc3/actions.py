"""Types d'interface identiques à ceux du SDK ARC-AGI-3 vus par le harnais.

GameAction : RESET, ACTION1..ACTION7 (ACTION6 = clic x,y).  Le harnais appelle
`GameAction[name]` ; ici l'énumération a les mêmes noms.
EtatJeu    : NOT_FINISHED / WIN / GAME_OVER (comme `obs.state.name`).
FrameData  : .frame (liste de grilles 64x64 d'entiers 0..15), .state,
             .levels_completed, .available_actions, .game_id, .win_levels.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any


class GameAction(enum.IntEnum):
    RESET = 0
    ACTION1 = 1
    ACTION2 = 2
    ACTION3 = 3
    ACTION4 = 4
    ACTION5 = 5
    ACTION6 = 6
    ACTION7 = 7

    def is_simple(self) -> bool:
        return self not in (GameAction.RESET, GameAction.ACTION6)

    def is_complex(self) -> bool:
        return self is GameAction.ACTION6


class EtatJeu(enum.Enum):
    NOT_FINISHED = "NOT_FINISHED"
    WIN = "WIN"
    GAME_OVER = "GAME_OVER"


GameState = EtatJeu


@dataclass
class FrameData:
    game_id: str
    frame: list
    state: EtatJeu
    levels_completed: int
    win_levels: int
    available_actions: list
    action_input: Any = None
    extra: dict = field(default_factory=dict)
