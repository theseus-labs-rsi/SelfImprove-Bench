"""Deterministic single-chapter deck-building benchmark and simulator."""

from __future__ import annotations

import argparse
import json
import random
import uuid
from collections import Counter
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Mapping


DEFAULT_DATA_FILE = Path(__file__).with_name("combat_data.json")
INITIAL_GOLD = 120
REFERENCE_BEAM_WIDTH = 80


class CombatError(ValueError):
    """Raised when task data or a plan violates the deterministic rules."""


@dataclass(frozen=True)
class Card:
    id: str
    name: str
    tier: int
    price: int
    cost: int
    damage: int
    block: int


@dataclass(frozen=True)
class Enemy:
    id: str
    name: str
    tier: int
    boss: bool
    max_hp: int
    reward_gold: int
    intent_cycle: tuple[tuple[str, int], ...]


def _integer(value: Any, label: str, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise CombatError(f"{label} must be an integer >= {minimum}")
    return value


def _cards_from_mapping(raw: Any, source: str) -> dict[str, Card]:
    if not isinstance(raw, Mapping) or not raw:
        raise CombatError(f"{source} must be a non-empty object")
    cards: dict[str, Card] = {}
    for card_id, record in raw.items():
        if not isinstance(card_id, str) or not card_id or not isinstance(record, Mapping):
            raise CombatError(f"{source} contains an invalid card")
        if record.get("id", card_id) != card_id:
            raise CombatError(f"card {card_id} has an inconsistent id")
        name = record.get("name")
        if not isinstance(name, str) or not name.strip():
            raise CombatError(f"card {card_id} has an invalid name")
        cards[card_id] = Card(
            id=card_id,
            name=name,
            tier=_integer(record.get("tier"), f"card {card_id} tier", 1),
            price=_integer(record.get("price"), f"card {card_id} price", 1),
            cost=_integer(record.get("cost"), f"card {card_id} cost", 0),
            damage=_integer(record.get("damage"), f"card {card_id} damage", 0),
            block=_integer(record.get("block"), f"card {card_id} block", 0),
        )
        if cards[card_id].damage == 0 and cards[card_id].block == 0:
            raise CombatError(f"card {card_id} must deal damage or grant block")
    return cards


def _enemies_from_mapping(raw: Any, source: str) -> dict[str, Enemy]:
    if not isinstance(raw, Mapping) or not raw:
        raise CombatError(f"{source} must be a non-empty object")
    enemies: dict[str, Enemy] = {}
    for enemy_id, record in raw.items():
        if not isinstance(enemy_id, str) or not enemy_id or not isinstance(record, Mapping):
            raise CombatError(f"{source} contains an invalid enemy")
        if record.get("id", enemy_id) != enemy_id:
            raise CombatError(f"enemy {enemy_id} has an inconsistent id")
        name = record.get("name")
        intents = record.get("intent_cycle")
        if not isinstance(name, str) or not name.strip():
            raise CombatError(f"enemy {enemy_id} has an invalid name")
        if not isinstance(record.get("boss"), bool):
            raise CombatError(f"enemy {enemy_id} boss must be boolean")
        if not isinstance(intents, list) or not intents:
            raise CombatError(f"enemy {enemy_id} intent_cycle must be non-empty")
        parsed_intents: list[tuple[str, int]] = []
        for intent in intents:
            if not isinstance(intent, Mapping) or intent.get("type") not in {"attack", "guard"}:
                raise CombatError(f"enemy {enemy_id} has an invalid intent")
            parsed_intents.append(
                (str(intent["type"]), _integer(intent.get("value"), "intent value", 1))
            )
        enemies[enemy_id] = Enemy(
            id=enemy_id,
            name=name,
            tier=_integer(record.get("tier"), f"enemy {enemy_id} tier", 1),
            boss=bool(record["boss"]),
            max_hp=_integer(record.get("max_hp"), f"enemy {enemy_id} max_hp", 1),
            reward_gold=_integer(
                record.get("reward_gold"), f"enemy {enemy_id} reward_gold", 0
            ),
            intent_cycle=tuple(parsed_intents),
        )
    return enemies


def load_combat_data(path: str | Path = DEFAULT_DATA_FILE) -> tuple[dict[str, Card], dict[str, Enemy]]:
    """Load and validate the global card and enemy catalog."""
    source = Path(path)
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CombatError(f"unable to read combat data: {source}") from exc
    if not isinstance(raw, Mapping):
        raise CombatError("combat data must be an object")
    return (
        _cards_from_mapping(raw.get("cards"), "cards"),
        _enemies_from_mapping(raw.get("enemies"), "enemies"),
    )


def save_json(value: Any, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _card_record(card: Card) -> dict[str, Any]:
    return asdict(card)


def _enemy_record(enemy: Enemy) -> dict[str, Any]:
    record = asdict(enemy)
    record["intent_cycle"] = [
        {"type": intent_type, "value": value}
        for intent_type, value in enemy.intent_cycle
    ]
    return record


def _build_candidate(
    difficulty: int,
    rng: random.Random,
    task_id: str,
    cards: Mapping[str, Card],
    enemies: Mapping[str, Enemy],
) -> dict[str, Any]:
    available_cards = sorted(card_id for card_id, card in cards.items() if card.tier <= difficulty)
    initial_deck: Counter[str] = Counter({"strike": 4, "guard": 4, "iron_wave": 1})
    for tier in range(2, difficulty + 1):
        tier_cards = sorted(card.id for card in cards.values() if card.tier == tier)
        initial_deck[rng.choice(tier_cards)] += 1

    selected_enemy_ids: list[str] = []
    stages: list[dict[str, Any]] = []
    for tier in range(1, difficulty + 1):
        normal_pool = sorted(
            enemy.id for enemy in enemies.values() if enemy.tier == tier and not enemy.boss
        )
        enemy_id = rng.choice(normal_pool)
        selected_enemy_ids.append(enemy_id)
        draw_order = list(available_cards)
        rng.shuffle(draw_order)
        stages.append(
            {
                "stage_id": f"combat_{tier}",
                "type": "combat",
                "enemy": enemy_id,
                "draw_order": draw_order,
            }
        )

        offer_max_tier = min(difficulty, tier + 1)
        offer_pool = [card_id for card_id in available_cards if cards[card_id].tier <= offer_max_tier]
        required_tier = offer_max_tier
        required = rng.choice(
            sorted(card_id for card_id in offer_pool if cards[card_id].tier == required_tier)
        )
        remaining = [card_id for card_id in offer_pool if card_id != required]
        offers = [required, *rng.sample(remaining, 2)]
        rng.shuffle(offers)
        stages.append(
            {
                "stage_id": f"shop_{tier}",
                "type": "shop",
                "offers": offers,
                "heal_price": 35,
                "heal_amount": 18,
            }
        )

    boss_pool = sorted(
        enemy.id for enemy in enemies.values() if enemy.tier == difficulty and enemy.boss
    )
    boss_id = rng.choice(boss_pool)
    selected_enemy_ids.append(boss_id)
    boss_draw_order = list(available_cards)
    rng.shuffle(boss_draw_order)
    stages.append(
        {
            "stage_id": "boss",
            "type": "combat",
            "enemy": boss_id,
            "draw_order": boss_draw_order,
        }
    )

    selected_enemies = {enemy_id: enemies[enemy_id] for enemy_id in sorted(set(selected_enemy_ids))}
    return {
        "task_id": task_id,
        "difficulty": difficulty,
        "player": {
            "max_hp": 80,
            "initial_hp": 80,
            "initial_gold": INITIAL_GOLD,
            "energy_per_turn": 3,
            "draw_count": 5,
        },
        "initial_deck": dict(sorted(initial_deck.items())),
        "card_catalog": {card_id: _card_record(cards[card_id]) for card_id in available_cards},
        "enemy_catalog": {
            enemy_id: _enemy_record(enemy) for enemy_id, enemy in selected_enemies.items()
        },
        "stages": stages,
        "rules": {
            "single_chapter": True,
            "maximum_card_tier": difficulty,
            "maximum_enemy_tier": difficulty,
            "max_combat_turns": 15,
            "player_block_resets_each_turn": True,
            "enemy_intents_are_public_and_repeat": True,
            "draw_order": (
                "For each combat, expand owned card counts in the public draw_order; "
                "copies are distributed round-robin, draw_count cards are drawn each turn, "
                "and the resulting cycle repeats."
            ),
            "shop_order": "buy listed cards in plan order, then optionally buy one heal",
        },
    }


def _task_parts(
    task: Mapping[str, Any],
) -> tuple[dict[str, Card], dict[str, Enemy], list[Mapping[str, Any]]]:
    if not isinstance(task, Mapping):
        raise CombatError("task must be an object")
    cards = _cards_from_mapping(task.get("card_catalog"), "task card_catalog")
    enemies = _enemies_from_mapping(task.get("enemy_catalog"), "task enemy_catalog")
    difficulty = _integer(task.get("difficulty"), "difficulty", 1)
    if difficulty > 5:
        raise CombatError("difficulty must be between 1 and 5")
    if any(card.tier > difficulty for card in cards.values()):
        raise CombatError("card_catalog contains a card above task difficulty")
    if any(enemy.tier > difficulty for enemy in enemies.values()):
        raise CombatError("enemy_catalog contains an enemy above task difficulty")
    stages = task.get("stages")
    if not isinstance(stages, list) or not stages:
        raise CombatError("stages must be a non-empty array")
    seen: set[str] = set()
    for stage in stages:
        if not isinstance(stage, Mapping) or not isinstance(stage.get("stage_id"), str):
            raise CombatError("each stage must have a string stage_id")
        stage_id = str(stage["stage_id"])
        if stage_id in seen:
            raise CombatError(f"duplicate stage_id: {stage_id}")
        seen.add(stage_id)
        stage_type = stage.get("type")
        if stage_type == "combat":
            if stage.get("enemy") not in enemies:
                raise CombatError(f"stage {stage_id} references an unknown enemy")
            order = stage.get("draw_order")
            if (
                not isinstance(order, list)
                or set(order) != set(cards)
                or len(order) != len(cards)
            ):
                raise CombatError(f"stage {stage_id} draw_order must list every card once")
        elif stage_type == "shop":
            offers = stage.get("offers")
            if (
                not isinstance(offers, list)
                or len(offers) != len(set(offers))
                or not all(card_id in cards for card_id in offers)
            ):
                raise CombatError(f"stage {stage_id} has invalid offers")
            _integer(stage.get("heal_price"), f"stage {stage_id} heal_price", 0)
            _integer(stage.get("heal_amount"), f"stage {stage_id} heal_amount", 1)
        else:
            raise CombatError(f"stage {stage_id} has an unsupported type")
    return cards, enemies, stages


def _player_rules(task: Mapping[str, Any]) -> tuple[int, int, int, int, int]:
    player = task.get("player")
    rules = task.get("rules")
    if not isinstance(player, Mapping) or not isinstance(rules, Mapping):
        raise CombatError("task must contain player and rules objects")
    max_hp = _integer(player.get("max_hp"), "max_hp", 1)
    initial_hp = _integer(player.get("initial_hp"), "initial_hp", 1)
    if initial_hp > max_hp:
        raise CombatError("initial_hp cannot exceed max_hp")
    return (
        max_hp,
        initial_hp,
        _integer(player.get("initial_gold"), "initial_gold", 0),
        _integer(player.get("energy_per_turn"), "energy_per_turn", 1),
        _integer(player.get("draw_count"), "draw_count", 1),
    )


def _initial_deck(task: Mapping[str, Any], cards: Mapping[str, Card]) -> Counter[str]:
    raw = task.get("initial_deck")
    if not isinstance(raw, Mapping) or not raw:
        raise CombatError("initial_deck must be a non-empty object")
    deck: Counter[str] = Counter()
    for card_id, count in raw.items():
        if card_id not in cards:
            raise CombatError(f"initial_deck contains unknown card: {card_id}")
        deck[str(card_id)] = _integer(count, f"initial_deck {card_id}", 1)
    return deck


def _deck_cycle(deck: Mapping[str, int], draw_order: Iterable[str]) -> list[str]:
    """Expand counts in a public, deterministic round-robin card order."""
    maximum = max(deck.values(), default=0)
    cycle: list[str] = []
    order = list(draw_order)
    for copy_index in range(maximum):
        cycle.extend(card_id for card_id in order if deck.get(card_id, 0) > copy_index)
    if not cycle:
        raise CombatError("deck cannot be empty")
    return cycle


def draw_hand(
    deck: Mapping[str, int], draw_order: Iterable[str], turn: int, draw_count: int
) -> list[str]:
    """Return the exact public hand for one 1-based combat turn."""
    if turn < 1:
        raise CombatError("turn must be positive")
    cycle = _deck_cycle(deck, draw_order)
    start = (turn - 1) * draw_count
    return [cycle[(start + offset) % len(cycle)] for offset in range(draw_count)]


def _action_options(hand: list[str], cards: Mapping[str, Card], energy: int) -> list[tuple[str, ...]]:
    options: dict[tuple[int, int, int], tuple[str, ...]] = {(0, 0, 0): ()}
    for count in range(1, len(hand) + 1):
        for indices in combinations(range(len(hand)), count):
            selected = tuple(hand[index] for index in indices)
            cost = sum(cards[card_id].cost for card_id in selected)
            if cost > energy:
                continue
            effect = (
                cost,
                sum(cards[card_id].damage for card_id in selected),
                sum(cards[card_id].block for card_id in selected),
            )
            current = options.get(effect)
            if current is None or selected < current:
                options[effect] = selected
    return sorted(options.values(), key=lambda action: (len(action), action))


def _resolve_turn(
    player_hp: int,
    enemy_hp: int,
    enemy_block: int,
    plays: Iterable[str],
    cards: Mapping[str, Card],
    enemy: Enemy,
    turn: int,
) -> tuple[int, int, int, int, list[dict[str, Any]]]:
    player_block = 0
    play_log: list[dict[str, Any]] = []
    for card_id in plays:
        card = cards[card_id]
        absorbed = min(enemy_block, card.damage)
        enemy_block -= absorbed
        hp_damage = card.damage - absorbed
        enemy_hp -= hp_damage
        player_block += card.block
        play_log.append(
            {
                "card": card_id,
                "cost": card.cost,
                "damage_to_hp": hp_damage,
                "damage_to_block": absorbed,
                "block_gained": card.block,
            }
        )
        if enemy_hp <= 0:
            return player_hp, 0, enemy_block, 0, play_log

    # Guard from the prior enemy turn protects against this player turn, then expires.
    enemy_block = 0
    intent_type, intent_value = enemy.intent_cycle[(turn - 1) % len(enemy.intent_cycle)]
    damage_received = 0
    if intent_type == "attack":
        damage_received = max(0, intent_value - player_block)
        player_hp -= damage_received
    else:
        enemy_block = intent_value
    return player_hp, enemy_hp, enemy_block, damage_received, play_log


def _combat_plan_map(stage_plan: Mapping[str, Any], max_turns: int) -> dict[int, list[str]]:
    turns = stage_plan.get("turns", [])
    if not isinstance(turns, list):
        raise CombatError(f"stage {stage_plan.get('stage_id')} turns must be an array")
    result: dict[int, list[str]] = {}
    for entry in turns:
        if not isinstance(entry, Mapping):
            raise CombatError("combat turn entries must be objects")
        turn = _integer(entry.get("turn"), "combat turn", 1)
        if turn > max_turns or turn in result:
            raise CombatError(f"invalid or duplicate combat turn: {turn}")
        plays = entry.get("play", [])
        if not isinstance(plays, list) or not all(isinstance(card_id, str) for card_id in plays):
            raise CombatError(f"turn {turn} play must be a string array")
        result[turn] = list(plays)
    return result


def _simulate_combat(
    *,
    stage: Mapping[str, Any],
    stage_plan: Mapping[str, Any],
    cards: Mapping[str, Card],
    enemy: Enemy,
    deck: Mapping[str, int],
    player_hp: int,
    energy: int,
    draw_count: int,
    max_turns: int,
) -> tuple[bool, int, list[dict[str, Any]]]:
    planned = _combat_plan_map(stage_plan, max_turns)
    enemy_hp = enemy.max_hp
    enemy_block = 0
    log: list[dict[str, Any]] = []
    for turn in range(1, max_turns + 1):
        hand = draw_hand(deck, stage["draw_order"], turn, draw_count)
        plays = planned.get(turn, [])
        remaining = Counter(hand)
        spent = 0
        for card_id in plays:
            if card_id not in cards:
                raise CombatError(f"stage {stage['stage_id']} uses unknown card: {card_id}")
            if remaining[card_id] <= 0:
                raise CombatError(
                    f"stage {stage['stage_id']} turn {turn}: {card_id} is not available in hand"
                )
            remaining[card_id] -= 1
            spent += cards[card_id].cost
            if spent > energy:
                raise CombatError(
                    f"stage {stage['stage_id']} turn {turn}: energy cost exceeds {energy}"
                )
        hp_before = player_hp
        enemy_hp_before = enemy_hp
        block_before = enemy_block
        player_hp, enemy_hp, enemy_block, damage_received, play_log = _resolve_turn(
            player_hp, enemy_hp, enemy_block, plays, cards, enemy, turn
        )
        intent_type, intent_value = enemy.intent_cycle[(turn - 1) % len(enemy.intent_cycle)]
        log.append(
            {
                "turn": turn,
                "hand": hand,
                "plays": play_log,
                "energy_spent": spent,
                "enemy_intent": {"type": intent_type, "value": intent_value},
                "player_hp_before": hp_before,
                "player_hp_after": max(0, player_hp),
                "enemy_hp_before": enemy_hp_before,
                "enemy_hp_after": max(0, enemy_hp),
                "enemy_block_before": block_before,
                "enemy_block_after": enemy_block,
                "damage_received": damage_received,
            }
        )
        if enemy_hp <= 0:
            return True, player_hp, log
        if player_hp <= 0:
            return False, 0, log
    return False, max(0, player_hp), log


def _ordered_plan_stages(
    task: Mapping[str, Any], plan: Mapping[str, Any], stages: list[Mapping[str, Any]]
) -> list[Mapping[str, Any]]:
    if not isinstance(plan, Mapping):
        raise CombatError("plan must be an object")
    if plan.get("task_id") not in (None, task.get("task_id")):
        raise CombatError("plan task_id does not match task")
    raw = plan.get("stages")
    if not isinstance(raw, list) or len(raw) != len(stages):
        raise CombatError("plan stages must contain every task stage in order")
    expected = [stage["stage_id"] for stage in stages]
    actual = [entry.get("stage_id") if isinstance(entry, Mapping) else None for entry in raw]
    if actual != expected:
        raise CombatError("plan stage IDs or order do not match the task")
    return list(raw)


def _objective_score(victory: bool, defeated: int, hp: int, gold: int) -> int:
    """Calculate the common final score for both successful and failed runs."""
    return 100 + defeated * 5 + max(0, hp) + gold


class CombatSession:
    """Execute one task stage at a time while exposing only current information."""

    def __init__(self, task: Mapping[str, Any]) -> None:
        self.task = task
        self.cards, self.enemies, self.stages = _task_parts(task)
        (
            self.max_hp,
            self.hp,
            self.gold,
            self.energy,
            self.draw_count,
        ) = _player_rules(task)
        self.deck = _initial_deck(task, self.cards)
        self.max_turns = _integer(
            task["rules"].get("max_combat_turns"), "max_combat_turns", 1
        )
        self.stage_index = 0
        self.defeated = 0
        self.failed_stage: str | None = None
        self.stage_log: list[dict[str, Any]] = []

    @property
    def finished(self) -> bool:
        return self.failed_stage is not None or self.stage_index >= len(self.stages)

    @property
    def current_stage(self) -> Mapping[str, Any] | None:
        if self.finished:
            return None
        return self.stages[self.stage_index]

    def current_request(self) -> dict[str, Any]:
        """Build the model-facing snapshot for the current stage only."""
        stage = self.current_stage
        if stage is None:
            raise CombatError("the combat session has already finished")
        stage_id = str(stage["stage_id"])
        request: dict[str, Any] = {
            "task_id": self.task.get("task_id"),
            "decision_type": stage["type"],
            "stage_id": stage_id,
            "progress": {
                "stage_number": self.stage_index + 1,
                "enemies_defeated": self.defeated,
            },
            "player_state": {
                "hp": self.hp,
                "max_hp": self.max_hp,
                "gold": self.gold,
                "energy_per_turn": self.energy,
                "draw_count": self.draw_count,
            },
            "deck": dict(sorted(self.deck.items())),
        }
        if stage["type"] == "combat":
            enemy = self.enemies[str(stage["enemy"])]
            visible_draw_order = [
                card_id for card_id in stage["draw_order"] if self.deck.get(card_id, 0) > 0
            ]
            owned_cards = {
                card_id: _card_record(self.cards[card_id])
                for card_id in sorted(self.deck)
            }
            draws = []
            for turn in range(1, self.max_turns + 1):
                intent_type, intent_value = enemy.intent_cycle[
                    (turn - 1) % len(enemy.intent_cycle)
                ]
                draws.append(
                    {
                        "turn": turn,
                        "hand": draw_hand(
                            self.deck, visible_draw_order, turn, self.draw_count
                        ),
                        "enemy_intent": {
                            "type": intent_type,
                            "value": intent_value,
                        },
                    }
                )
            request.update(
                {
                    "card_catalog": owned_cards,
                    "combat": {
                        "enemy": _enemy_record(enemy),
                        "draw_order": visible_draw_order,
                        "draw_cycle": _deck_cycle(self.deck, visible_draw_order),
                        "draws": draws,
                        "max_turns": self.max_turns,
                    },
                    "rules": {
                        "player_block_resets_each_turn": True,
                        "played_cards_resolve_left_to_right": True,
                        "enemy_block_expires_before_its_next_intent": True,
                    },
                }
            )
            return request

        offer_ids = list(stage["offers"])
        visible_card_ids = sorted(set(self.deck) | set(offer_ids))
        request.update(
            {
                "card_catalog": {
                    card_id: _card_record(self.cards[card_id])
                    for card_id in visible_card_ids
                },
                "shop": {
                    "offers": [
                        _card_record(self.cards[card_id]) for card_id in offer_ids
                    ],
                    "heal_price": int(stage["heal_price"]),
                    "heal_amount": int(stage["heal_amount"]),
                    "purchase_order": "buy_cards in listed order, then heal",
                }
            }
        )
        return request

    def apply(self, stage_plan: Mapping[str, Any]) -> dict[str, Any]:
        """Validate and apply one current-stage decision, updating persistent state."""
        stage = self.current_stage
        if stage is None:
            raise CombatError("the combat session has already finished")
        if not isinstance(stage_plan, Mapping):
            raise CombatError("stage plan must be an object")
        if stage_plan.get("task_id") not in (None, self.task.get("task_id")):
            raise CombatError("stage plan task_id does not match task")
        stage_id = str(stage["stage_id"])
        if stage_plan.get("stage_id") != stage_id:
            raise CombatError(f"expected a decision for stage {stage_id}")

        if stage["type"] == "combat":
            enemy = self.enemies[str(stage["enemy"])]
            won, self.hp, combat_log = _simulate_combat(
                stage=stage,
                stage_plan=stage_plan,
                cards=self.cards,
                enemy=enemy,
                deck=self.deck,
                player_hp=self.hp,
                energy=self.energy,
                draw_count=self.draw_count,
                max_turns=self.max_turns,
            )
            reward = enemy.reward_gold if won else 0
            self.gold += reward
            record = {
                "stage_id": stage_id,
                "type": "combat",
                "enemy": enemy.id,
                "won": won,
                "reward_gold": reward,
                "player_hp": self.hp,
                "gold": self.gold,
                "turns": combat_log,
            }
            self.stage_log.append(record)
            self.stage_index += 1
            if won:
                self.defeated += 1
            else:
                self.failed_stage = stage_id
            return record

        purchases = stage_plan.get("buy_cards", [])
        heal = stage_plan.get("heal", False)
        if not isinstance(purchases, list) or not all(
            isinstance(card_id, str) for card_id in purchases
        ):
            raise CombatError(f"stage {stage_id} buy_cards must be a string array")
        if len(purchases) != len(set(purchases)):
            raise CombatError(f"stage {stage_id} cannot buy an offered card twice")
        if not isinstance(heal, bool):
            raise CombatError(f"stage {stage_id} heal must be boolean")
        offered = set(stage["offers"])
        bought: list[dict[str, Any]] = []
        for card_id in purchases:
            if card_id not in offered:
                raise CombatError(f"stage {stage_id} card is not offered: {card_id}")
            price = self.cards[card_id].price
            if self.gold < price:
                raise CombatError(f"stage {stage_id} cannot afford {card_id}")
            self.gold -= price
            self.deck[card_id] += 1
            bought.append({"card": card_id, "price": price})
        healed = 0
        if heal:
            heal_price = int(stage["heal_price"])
            if self.gold < heal_price:
                raise CombatError(f"stage {stage_id} cannot afford healing")
            self.gold -= heal_price
            healed = min(int(stage["heal_amount"]), self.max_hp - self.hp)
            self.hp += healed
        record = {
            "stage_id": stage_id,
            "type": "shop",
            "bought": bought,
            "heal_purchased": heal,
            "hp_restored": healed,
            "player_hp": self.hp,
            "gold": self.gold,
            "deck": dict(sorted(self.deck.items())),
        }
        self.stage_log.append(record)
        self.stage_index += 1
        return record

    def result(self, *, include_reference: bool = True) -> dict[str, Any]:
        total_combats = sum(stage["type"] == "combat" for stage in self.stages)
        victory = self.defeated == total_combats
        result: dict[str, Any] = {
            "task_id": self.task.get("task_id"),
            "victory": victory,
            "failed_stage": self.failed_stage,
            "enemies_defeated": self.defeated,
            "total_enemies": total_combats,
            "stages_reached": len(self.stage_log),
            "total_stages": len(self.stages),
            "final_hp": self.hp,
            "final_gold": self.gold,
            "final_deck": dict(sorted(self.deck.items())),
            "objective_score": _objective_score(
                victory, self.defeated, self.hp, self.gold
            ),
            "stage_log": self.stage_log,
        }
        if include_reference:
            reference = self.task.get("reference_score")
            if reference is not None:
                reference = _integer(reference, "reference_score", 1)
                result["reference_score"] = reference
                result["relative_score"] = result["objective_score"] / reference
                result["score"] = result["relative_score"]
            else:
                result["reference_score"] = None
                result["relative_score"] = None
                result["score"] = None
        return result


def _simulate_core(task: Mapping[str, Any], plan: Mapping[str, Any]) -> dict[str, Any]:
    session = CombatSession(task)
    stages = session.stages
    plan_stages = _ordered_plan_stages(task, plan, stages)
    for stage_plan in plan_stages:
        if session.finished:
            break
        session.apply(stage_plan)
    return session.result(include_reference=False)


def simulate_plan(task: Mapping[str, Any], plan: Mapping[str, Any]) -> dict[str, Any]:
    """Execute a legacy complete plan or a finished sequential-stage plan."""
    session = CombatSession(task)
    if not isinstance(plan, Mapping):
        raise CombatError("plan must be an object")
    sequential = plan.get("mode") == "sequential_stage_decisions"
    if sequential:
        if plan.get("task_id") not in (None, task.get("task_id")):
            raise CombatError("plan task_id does not match task")
        plan_stages = plan.get("stages")
        if not isinstance(plan_stages, list):
            raise CombatError("sequential plan stages must be an array")
    else:
        plan_stages = _ordered_plan_stages(task, plan, session.stages)
    for stage_plan in plan_stages:
        if session.finished:
            if sequential:
                raise CombatError("plan contains decisions after the run has finished")
            break
        session.apply(stage_plan)
    if sequential and not session.finished:
        raise CombatError("sequential plan ends before the run has finished")
    result = session.result()
    return result


def _best_combat_plan(
    *,
    stage: Mapping[str, Any],
    cards: Mapping[str, Card],
    enemy: Enemy,
    deck: Mapping[str, int],
    player_hp: int,
    energy: int,
    draw_count: int,
    max_turns: int,
) -> tuple[int, list[dict[str, Any]]] | None:
    # State key is (enemy HP, enemy block); higher player HP dominates lower HP.
    states: dict[tuple[int, int], tuple[int, list[dict[str, Any]]]] = {
        (enemy.max_hp, 0): (player_hp, [])
    }
    winners: list[tuple[int, list[dict[str, Any]]]] = []
    for turn in range(1, max_turns + 1):
        hand = draw_hand(deck, stage["draw_order"], turn, draw_count)
        actions = _action_options(hand, cards, energy)
        next_states: dict[tuple[int, int], tuple[int, list[dict[str, Any]]]] = {}
        for (enemy_hp, enemy_block), (hp, history) in states.items():
            for action in actions:
                next_hp, next_enemy_hp, next_enemy_block, _, _ = _resolve_turn(
                    hp, enemy_hp, enemy_block, action, cards, enemy, turn
                )
                next_history = [*history, {"turn": turn, "play": list(action)}]
                if next_enemy_hp <= 0:
                    winners.append((next_hp, next_history))
                    continue
                if next_hp <= 0:
                    continue
                key = (next_enemy_hp, next_enemy_block)
                current = next_states.get(key)
                if current is None or next_hp > current[0]:
                    next_states[key] = (next_hp, next_history)
        states = next_states
        if not states and winners:
            break
    if not winners:
        return None
    return max(winners, key=lambda item: (item[0], -len(item[1])))


def _shop_successors(
    state: dict[str, Any],
    stage: Mapping[str, Any],
    cards: Mapping[str, Card],
    max_hp: int,
) -> list[dict[str, Any]]:
    offers = list(stage["offers"])
    successors: list[dict[str, Any]] = []
    for count in range(len(offers) + 1):
        for selected in combinations(offers, count):
            card_cost = sum(cards[card_id].price for card_id in selected)
            for heal in (False, True):
                cost = card_cost + (int(stage["heal_price"]) if heal else 0)
                if cost > state["gold"]:
                    continue
                deck = Counter(state["deck"])
                deck.update(selected)
                hp = state["hp"]
                if heal:
                    hp = min(max_hp, hp + int(stage["heal_amount"]))
                successors.append(
                    {
                        "hp": hp,
                        "gold": state["gold"] - cost,
                        "deck": dict(deck),
                        "plans": [
                            *state["plans"],
                            {
                                "stage_id": stage["stage_id"],
                                "buy_cards": list(selected),
                                "heal": heal,
                            },
                        ],
                        "defeated": state["defeated"],
                    }
                )
    return successors


def _deck_power(deck: Mapping[str, int], cards: Mapping[str, Card]) -> float:
    return sum(
        count * (card.damage + card.block) / max(1, card.cost)
        for card_id, count in deck.items()
        for card in (cards[card_id],)
    )


def solve_task(task: Mapping[str, Any], beam_width: int = REFERENCE_BEAM_WIDTH) -> dict[str, Any]:
    """Find and return a verified reference plan with deterministic beam search."""
    if beam_width < 1:
        raise CombatError("beam_width must be positive")
    cards, enemies, stages = _task_parts(task)
    max_hp, initial_hp, initial_gold, energy, draw_count = _player_rules(task)
    initial_deck = _initial_deck(task, cards)
    max_turns = _integer(task["rules"].get("max_combat_turns"), "max_combat_turns", 1)
    states: list[dict[str, Any]] = [
        {
            "hp": initial_hp,
            "gold": initial_gold,
            "deck": dict(initial_deck),
            "plans": [],
            "defeated": 0,
        }
    ]
    combat_cache: dict[tuple[Any, ...], tuple[int, list[dict[str, Any]]] | None] = {}

    for stage in stages:
        if stage["type"] == "combat":
            enemy = enemies[str(stage["enemy"])]
            survivors: list[dict[str, Any]] = []
            for state in states:
                deck_key = tuple(sorted(state["deck"].items()))
                key = (stage["stage_id"], state["hp"], deck_key)
                if key not in combat_cache:
                    combat_cache[key] = _best_combat_plan(
                        stage=stage,
                        cards=cards,
                        enemy=enemy,
                        deck=state["deck"],
                        player_hp=state["hp"],
                        energy=energy,
                        draw_count=draw_count,
                        max_turns=max_turns,
                    )
                outcome = combat_cache[key]
                if outcome is None:
                    continue
                hp, turns = outcome
                survivors.append(
                    {
                        "hp": hp,
                        "gold": state["gold"] + enemy.reward_gold,
                        "deck": state["deck"],
                        "plans": [
                            *state["plans"],
                            {"stage_id": stage["stage_id"], "turns": turns},
                        ],
                        "defeated": state["defeated"] + 1,
                    }
                )
            states = survivors
        else:
            states = [
                successor
                for state in states
                for successor in _shop_successors(state, stage, cards, max_hp)
            ]
        if not states:
            raise CombatError(f"reference solver cannot pass stage {stage['stage_id']}")

        unique: dict[tuple[Any, ...], dict[str, Any]] = {}
        for state in states:
            key = (state["hp"], state["gold"], tuple(sorted(state["deck"].items())))
            unique[key] = state
        states = sorted(
            unique.values(),
            key=lambda state: (
                state["hp"] * 100
                + state["gold"]
                + _deck_power(state["deck"], cards) * 25,
                state["gold"],
            ),
            reverse=True,
        )[:beam_width]

    best = max(
        states,
        key=lambda state: _objective_score(
            True, state["defeated"], state["hp"], state["gold"]
        ),
    )
    return {"task_id": task.get("task_id"), "stages": best["plans"]}


def generate_task(
    difficulty: int,
    data_file: str | Path = DEFAULT_DATA_FILE,
    seed: int | None = None,
    task_id: str | None = None,
    max_generation_attempts: int = 30,
) -> dict[str, Any]:
    """Generate a reproducible task and reject candidates without a verified solution."""
    difficulty = _integer(difficulty, "difficulty", 1)
    if difficulty > 5:
        raise CombatError("difficulty must be between 1 and 5")
    if max_generation_attempts < 1:
        raise CombatError("max_generation_attempts must be positive")
    cards, enemies = load_combat_data(data_file)
    for tier in range(1, difficulty + 1):
        if not any(card.tier == tier for card in cards.values()):
            raise CombatError(f"card catalog has no tier {tier} cards")
        if not any(enemy.tier == tier and not enemy.boss for enemy in enemies.values()):
            raise CombatError(f"enemy catalog has no normal tier {tier} enemy")
    if not any(enemy.tier == difficulty and enemy.boss for enemy in enemies.values()):
        raise CombatError(f"enemy catalog has no tier {difficulty} boss")

    rng = random.Random(seed)
    resolved_id = task_id or f"combat-{uuid.uuid4().hex[:10]}"
    last_error: Exception | None = None
    for attempt in range(1, max_generation_attempts + 1):
        candidate = _build_candidate(difficulty, rng, resolved_id, cards, enemies)
        try:
            reference_plan = solve_task(candidate)
            reference_result = _simulate_core(candidate, reference_plan)
        except CombatError as exc:
            last_error = exc
            continue
        if not reference_result["victory"]:
            last_error = CombatError("reference plan failed replay validation")
            continue
        candidate["reference_score"] = reference_result["objective_score"]
        candidate["reference_method"] = "deterministic_beam_search_verified"
        candidate["solvability"] = {
            "verified": True,
            "generation_attempt": attempt,
            "method": "reference plan replay through the public simulator",
            "beam_width": REFERENCE_BEAM_WIDTH,
        }
        return candidate
    raise CombatError(
        f"unable to generate a verified solvable task after {max_generation_attempts} attempts: "
        f"{last_error}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and simulate deterministic combat tasks")
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate = subparsers.add_parser("generate", help="generate one verified task")
    generate.add_argument("difficulty", type=int)
    generate.add_argument("--seed", type=int)
    generate.add_argument("--task-id")
    generate.add_argument("--data-file", default=str(DEFAULT_DATA_FILE))
    generate.add_argument("-o", "--output", default="task.json")
    simulate = subparsers.add_parser("simulate", help="simulate a plan")
    simulate.add_argument("task")
    simulate.add_argument("plan")
    simulate.add_argument("-o", "--output", default="result.json")
    solve = subparsers.add_parser("solve", help="write the reference solver plan")
    solve.add_argument("task")
    solve.add_argument("-o", "--output", default="reference_plan.json")
    args = parser.parse_args()
    try:
        if args.command == "generate":
            value = generate_task(
                args.difficulty,
                data_file=args.data_file,
                seed=args.seed,
                task_id=args.task_id,
            )
        else:
            task = json.loads(Path(args.task).read_text(encoding="utf-8"))
            if args.command == "simulate":
                plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
                value = simulate_plan(task, plan)
            else:
                value = solve_task(task)
        save_json(value, args.output)
        print(json.dumps(value, ensure_ascii=False, indent=2))
    except (CombatError, OSError, json.JSONDecodeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
