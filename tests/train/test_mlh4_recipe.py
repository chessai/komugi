"""The ``mlh4`` arm: KOM-159 §6's moves-left head at loss weight 4.0.

The arm is a *new* constant and a new name, never an edit to an existing one --
the curves campaigns 1-5 produced belong to ``P0B_RECIPE`` and campaign 6's
first 25 iterations belong to ``MLH_RECIPE``, so T2 pins both against an arm
that reached for `dataclasses.replace`'s target instead of its copy.
"""

from __future__ import annotations

import dataclasses

import pytest

from komugi.campaign.loop import (
    LOSS_RECIPE_IS_MLH,
    LOSS_RECIPES,
    RECIPE_MLH,
    RECIPE_MLH4,
    RECIPE_P0B,
    CampaignConfig,
    recipe_is_mlh,
)
from komugi.train.wiring import MLH4_RECIPE, MLH_RECIPE, P0B_RECIPE


def test_mlh4_is_mlh_with_the_head_at_weight_four() -> None:
    """T1: field for field ``MLH_RECIPE``, with ``mlh_weight`` 4.0."""
    assert MLH4_RECIPE == dataclasses.replace(MLH_RECIPE, mlh_weight=4.0)
    assert MLH4_RECIPE.mlh_weight == 4.0
    assert MLH4_RECIPE.mlh_enabled is True


def test_existing_recipes_are_unedited() -> None:
    """T2: adding the arm did not move the constants campaigns already ran."""
    assert MLH_RECIPE.mlh_weight == 1.0
    assert MLH_RECIPE.mlh_enabled is True
    assert P0B_RECIPE.mlh_enabled is False
    assert P0B_RECIPE.mlh_weight == 0.0


def test_loss_recipes_resolves_both_names() -> None:
    """T3: the closed mapping carries the new name and keeps the old one."""
    assert LOSS_RECIPES[RECIPE_MLH4] is MLH4_RECIPE
    assert LOSS_RECIPES[RECIPE_MLH] is MLH_RECIPE
    assert LOSS_RECIPES[RECIPE_P0B] is P0B_RECIPE
    assert RECIPE_MLH4 == "mlh4"


def _campaign(loss_recipe: str) -> CampaignConfig:
    """The fields the launcher passes, with the recipe under test."""
    return CampaignConfig(
        iterations=1,
        run_seed=0,
        root="/tmp/kom159d",
        steps_per_iteration=1,
        batch_size=1,
        lr=1e-3,
        workers=1,
        games_per_worker=1,
        device="cpu",
        loss_recipe=loss_recipe,
    )


def test_campaign_config_accepts_mlh4_and_rejects_unknown() -> None:
    """T4: validated by name, before anything spawns."""
    assert _campaign("mlh4").loss_recipe == RECIPE_MLH4
    with pytest.raises(ValueError, match="loss_recipe"):
        _campaign("mlh9")


@pytest.mark.parametrize(
    ("loss_recipe", "flag"),
    [(RECIPE_MLH, 1.0), (RECIPE_MLH4, 1.0), (RECIPE_P0B, 0.0)],
)
def test_journal_flag_is_the_moves_left_family(
    loss_recipe: str, flag: float
) -> None:
    """T5: ``loss_recipe_is_mlh`` covers both weights of the head."""
    assert float(recipe_is_mlh(loss_recipe)) == flag
    assert LOSS_RECIPE_IS_MLH == "loss_recipe_is_mlh"
