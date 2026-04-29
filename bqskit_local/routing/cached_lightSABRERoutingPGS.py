from __future__ import annotations

import logging
from typing import Sequence

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit

from bqskit_local.mapping.cached_lightSABRE_pgs import (
    GeneralizedCachedLightSABREAlgorithmPGS,
)
from bqskit_local.mapping.lightSABRE_pgs import DEFAULT_LIGHTSABRE_HEURISTIC
from bqskit_local.position.graph import PositionGraph
from bqskit_local.position.state import PositionGraphState
from bqskit_local.routing.lightSABRERoutingPGS import (
    GeneralizedLightSABRERoutingPassPGS,
)

_logger = logging.getLogger(__name__)


class GeneralizedCachedLightSABRERoutingPassPGS(
    BasePass,
    GeneralizedCachedLightSABREAlgorithmPGS,
):
    """Cached LightSABRE-style PGS routing with multiple seeded trials."""

    def __init__(
        self,
        template_pgs: PositionGraphState,
        decay_delta: float = 0.001,
        decay_reset_interval: int = 5,
        decay_reset_on_gate: bool = True,
        extended_set_size: int = 20,
        extended_set_weight: float = 0.5,
        cg_compatibility_mode: bool = False,
        heuristic: str = DEFAULT_LIGHTSABRE_HEURISTIC,
        seed: int | None = None,
        trials: int = 1,
        attempt_limit: int | None = None,
    ) -> None:
        if not isinstance(template_pgs, PositionGraphState):
            raise TypeError(
                f'Expected PositionGraphState, got {type(template_pgs)}.',
            )

        if not isinstance(trials, int):
            raise TypeError(f'Expected int for trials, got {type(trials)}.')

        if trials < 1:
            raise ValueError('trials must be a positive integer.')

        self.template_pgs = template_pgs.copy()
        self.trials = trials

        super().__init__(
            decay_delta=decay_delta,
            decay_reset_interval=decay_reset_interval,
            decay_reset_on_gate=decay_reset_on_gate,
            extended_set_size=extended_set_size,
            extended_set_weight=extended_set_weight,
            cg_compatibility_mode=cg_compatibility_mode,
            heuristic=heuristic,
            seed=seed,
            attempt_limit=attempt_limit,
        )

    _build_local_pgs = GeneralizedLightSABRERoutingPassPGS._build_local_pgs
    _build_compatibility_local_pgs = (
        GeneralizedLightSABRERoutingPassPGS._build_compatibility_local_pgs
    )
    run = GeneralizedLightSABRERoutingPassPGS.run
