# Deterministic Combat Stage Decision

You control one stage of a deterministic deck-building run. The caller advances the game after every response and will invoke you again only if the run reaches another stage. Make a legal decision for the current stage only. Do not invent or plan later stages.

The run score is `100 + 5 * enemies_defeated + final_hp + final_gold`; victory over the current enemy is required to continue. Within the information available now, preserve HP and gold while making choices that improve the chance of reaching and defeating the Boss.

Use a short, bounded calculation. Do not perform exhaustive search or explain your reasoning. Return the JSON decision immediately after one legality check.

## Input

Every input contains:

- `task_id`, `stage_id`, and `decision_type` (`combat` or `shop`).
- `player_state`: current HP, maximum HP, gold, energy, and draw count.
- `deck`: the currently owned card counts. HP, gold, and deck changes persist.

A combat input additionally contains:

- `card_catalog`: stats for every currently owned card.
- `combat.enemy`: this enemy's HP, reward, and repeating public intent cycle.
- `combat.draw_cycle`: the effective deterministic deck cycle.
- `combat.draws`: the exact hand and enemy intent for every allowed turn.
- `combat.max_turns`: the final turn by which the enemy must be defeated.

A shop input additionally contains `shop.offers` with complete card stats and prices, plus the heal price and amount. No later enemy or shop information is available.

## Combat rules

- Choose cards only from that turn's `hand`. A card ID cannot be played more times than it occurs in the hand.
- Total cost per turn cannot exceed `player_state.energy_per_turn`.
- Cards resolve left to right. Damage removes enemy block before HP; block reduces the surviving enemy's attack that turn.
- Player block resets every turn. Enemy guard supplies block for the next player turn, then expires before the enemy's following intent.
- If the enemy reaches 0 HP while cards are resolving, it dies immediately and does not act.
- Preserve HP where practical, but defeating the enemy within `max_turns` is the primary goal.

For combat, return exactly one object of this shape:

```json
{
  "task_id": "combat-example",
  "stage_id": "combat_1",
  "turns": [
    {"turn": 1, "play": ["strike", "guard"]},
    {"turn": 2, "play": ["iron_wave", "strike"]}
  ]
}
```

Omitted turns play no cards. Turn numbers must be unique positive integers no greater than `combat.max_turns`.

## Shop rules

- Buy only IDs in `shop.offers`, at most once each.
- Purchases execute in `buy_cards` order. Their total cost plus healing cannot exceed current gold.
- Healing executes after card purchases, costs `heal_price`, and restores at most `heal_amount` up to maximum HP.
- Balance useful deck improvement against retaining gold. Do not assume knowledge of future stages.

For a shop, return exactly one object of this shape:

```json
{
  "task_id": "combat-example",
  "stage_id": "shop_1",
  "buy_cards": ["pommel"],
  "heal": false
}
```

Return only the applicable JSON object, without Markdown fences, commentary, future-stage plans, or extra narrative fields.
