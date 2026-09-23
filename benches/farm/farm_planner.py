"""Text based farm planning task generator and simulator.

The module has no third-party dependencies and can be used both as a library
and from the command line.  A plan is executed in day order, with operations
on one day ordered as buy -> plant -> harvest -> sell.
"""

from __future__ import annotations

import argparse
import json
import random
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


DEFAULT_CROP_FILE = Path(__file__).with_name("crop_data.json")


class FarmError(ValueError):
    """Raised when a task or plan is malformed or violates game rules."""


@dataclass(frozen=True)
class Crop:
    id: str
    name: str
    seed_price: int
    maturity_days: int
    yield_per_harvest: int
    fruit_price: int


def _crops_from_mapping(raw: Any, source: str) -> dict[str, Crop]:
    """Validate a crop mapping and convert it to :class:`Crop` objects."""
    if not isinstance(raw, dict) or not raw:
        raise FarmError(f"{source} must be a non-empty JSON object")
    fields = ("name", "seed_price", "maturity_days", "yield_per_harvest", "fruit_price")
    crops: dict[str, Crop] = {}
    for crop_id, values in raw.items():
        if not isinstance(crop_id, str) or not crop_id or not isinstance(values, dict):
            raise FarmError("crop IDs and records must be objects")
        if any(field not in values for field in fields):
            raise FarmError(f"crop {crop_id} is missing a required field")
        if "id" in values and values["id"] != crop_id:
            raise FarmError(f"crop {crop_id} has an inconsistent id")
        if not isinstance(values["name"], str) or not values["name"].strip():
            raise FarmError(f"crop {crop_id} name must be a non-empty string")
        if (
            not isinstance(values["seed_price"], int)
            or isinstance(values["seed_price"], bool)
            or values["seed_price"] <= 0
        ) or any(
            not isinstance(values[field], int)
            or isinstance(values[field], bool)
            or values[field] <= 0
            for field in fields[2:]
        ):
            raise FarmError(f"crop {crop_id} numeric fields must be positive integers")
        crops[crop_id] = Crop(id=crop_id, **{field: values[field] for field in fields})
    return crops


def load_crops(path: str | Path = DEFAULT_CROP_FILE) -> dict[str, Crop]:
    """Load and validate the global crop JSON."""
    source = Path(path)
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FarmError(f"无法读取作物数据: {source}") from exc
    if not isinstance(raw, dict) or not raw:
        raise FarmError("作物数据必须是非空 JSON 对象")
    crops: dict[str, Crop] = {}
    fields = ("name", "seed_price", "maturity_days", "yield_per_harvest", "fruit_price")
    for crop_id, values in raw.items():
        if not isinstance(crop_id, str) or not crop_id or not isinstance(values, dict):
            raise FarmError("作物 ID 和作物记录必须是对象")
        if any(field not in values for field in fields):
            raise FarmError(f"作物 {crop_id} 缺少基础字段")
        if not isinstance(values["name"], str) or not values["name"].strip():
            raise FarmError(f"作物 {crop_id} 的 name 必须是非空字符串")
        if not isinstance(values["seed_price"], int) or isinstance(values["seed_price"], bool) or values["seed_price"] <= 0:
            raise FarmError(f"作物 {crop_id} 的 seed_price 必须是非负整数")
        if any(not isinstance(values[field], int) or isinstance(values[field], bool) or values[field] <= 0 for field in fields[2:]):
            raise FarmError(f"作物 {crop_id} 的数值字段必须是正整数")
        crops[crop_id] = Crop(id=crop_id, **{field: values[field] for field in fields})
    return crops


def _positive_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise FarmError(f"{label} 必须是正整数")
    return value


def _nonnegative_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise FarmError(f"{label} must be a non-negative integer")
    return value


def generate_task(
    difficulty: int,
    crop_file: str | Path = DEFAULT_CROP_FILE,
    seed: int | None = None,
    task_id: str | None = None,
) -> dict[str, Any]:
    """Generate a reproducible task case for a difficulty coefficient."""
    difficulty = _nonnegative_int(difficulty, "difficulty")
    crops = load_crops(crop_file)
    available_count = difficulty * 2 + 2
    if available_count > len(crops):
        raise FarmError(f"难度 {difficulty} 需要 {available_count} 种作物，但数据中只有 {len(crops)} 种")
    rng = random.Random(seed)
    selected = sorted(rng.sample(list(crops), available_count))
    task = {
        "task_id": task_id or f"farm-{uuid.uuid4().hex[:10]}",
        "difficulty": difficulty,
        "initial_money": 200,
        "farmland_count": difficulty * 4 + 4,
        "duration_days": difficulty * 4 + 4,
        "available_crops": selected,
        "crop_catalog": {crop_id: asdict(crops[crop_id]) | {"id": crop_id} for crop_id in selected},
        "rules": {
            "day_numbering": "1-based",
            "daily_operation_order": ["buy_seeds", "plant", "harvest", "sell"],
            "harvest_day": "plant_day + maturity_days",
            "unsold_fruit_has_no_value": True,
        },
    }
    task["reference_daily_profits"] = {
        crop_id: (crops[crop_id].yield_per_harvest * crops[crop_id].fruit_price - crops[crop_id].seed_price)
        / crops[crop_id].maturity_days
        for crop_id in selected
    }
    task["reference_profit"] = calculate_reference_profit(task)
    task["reference_method"] = "greedy_daily_profit_lookahead"
    return task


def _greedy_reference_profit(
    initial_money: int,
    farmland: int,
    duration: int,
    crops: dict[str, Crop],
    available: Iterable[str],
) -> int:
    """Calculate a deterministic greedy benchmark with one-step lookahead.

    A purely local daily-profit choice can spend all cash on a long crop and
    miss several profitable short cycles. For every purchase opportunity we
    therefore project the terminal cash for each currently affordable crop,
    filling the free plots with that crop and using the local daily-profit
    greedy rule for the remaining days. This keeps the reference heuristic
    deterministic while avoiding that pathological short-sighted behavior.
    """
    ranked = sorted(
        (crops[crop_id] for crop_id in available),
        key=lambda crop: (
            (crop.yield_per_harvest * crop.fruit_price - crop.seed_price) / crop.maturity_days,
            crop.id,
        ),
        reverse=True,
    )

    def run_daily_greedy(start_day: int, start_money: int, start_growing: list[dict[str, int | str]]) -> int:
        """Replay the simple daily-profit greedy policy from an intermediate state."""
        money = start_money
        growing = [dict(batch) for batch in start_growing]
        for day in range(start_day, duration + 1):
            remaining: list[dict[str, int | str]] = []
            for batch in growing:
                if int(batch["harvest_day"]) <= day:
                    crop = crops[str(batch["crop"])]
                    money += int(batch["quantity"]) * crop.yield_per_harvest * crop.fruit_price
                else:
                    remaining.append(batch)
            growing = remaining
            while len(growing) < farmland:
                affordable = next(
                    (
                        crop
                        for crop in ranked
                        if crop.seed_price <= money and day + crop.maturity_days <= duration
                    ),
                    None,
                )
                if affordable is None:
                    break
                quantity = min(farmland - len(growing), money // affordable.seed_price)
                if quantity <= 0:
                    break
                money -= quantity * affordable.seed_price
                growing.extend(
                    {"crop": affordable.id, "quantity": 1, "harvest_day": day + affordable.maturity_days}
                    for _ in range(quantity)
                )
        return money

    money = initial_money
    growing: list[dict[str, int | str]] = []
    for day in range(1, duration + 1):
        remaining: list[dict[str, int | str]] = []
        for batch in growing:
            if int(batch["harvest_day"]) <= day:
                crop = crops[str(batch["crop"])]
                money += int(batch["quantity"]) * crop.yield_per_harvest * crop.fruit_price
            else:
                remaining.append(batch)
        growing = remaining
        while len(growing) < farmland:
            candidates = [
                crop
                for crop in ranked
                if crop.seed_price <= money and day + crop.maturity_days <= duration
            ]
            if not candidates:
                break
            free_plots = farmland - len(growing)

            def projected_terminal_cash(crop: Crop) -> int:
                quantity = min(free_plots, money // crop.seed_price)
                projected_money = money - quantity * crop.seed_price
                projected_growing = [dict(batch) for batch in growing]
                projected_growing.extend(
                    {"crop": crop.id, "quantity": 1, "harvest_day": day + crop.maturity_days}
                    for _ in range(quantity)
                )
                # The real policy keeps buying on the same day while cash and
                # plots remain. Mirror that fallback purchase in the
                # projection before replaying later days.
                while len(projected_growing) < farmland:
                    fallback = next(
                        (
                            candidate
                            for candidate in ranked
                            if candidate.seed_price <= projected_money
                            and day + candidate.maturity_days <= duration
                        ),
                        None,
                    )
                    if fallback is None:
                        break
                    fallback_quantity = min(
                        farmland - len(projected_growing),
                        projected_money // fallback.seed_price,
                    )
                    if fallback_quantity <= 0:
                        break
                    projected_money -= fallback_quantity * fallback.seed_price
                    projected_growing.extend(
                        {
                            "crop": fallback.id,
                            "quantity": 1,
                            "harvest_day": day + fallback.maturity_days,
                        }
                        for _ in range(fallback_quantity)
                    )
                return run_daily_greedy(day + 1, projected_money, projected_growing)

            affordable = max(
                candidates,
                key=lambda crop: (
                    projected_terminal_cash(crop),
                    (crop.yield_per_harvest * crop.fruit_price - crop.seed_price) / crop.maturity_days,
                    crop.id,
                ),
            )
            quantity = min(free_plots, money // affordable.seed_price)
            if quantity <= 0:
                break
            money -= quantity * affordable.seed_price
            growing.extend(
                {"crop": affordable.id, "quantity": 1, "harvest_day": day + affordable.maturity_days}
                for _ in range(quantity)
            )
    return money - initial_money


def calculate_reference_profit(task: dict[str, Any], crop_file: str | Path = DEFAULT_CROP_FILE) -> int:
    """Return the greedy benchmark profit for a generated or compatible task."""
    if not isinstance(task, dict):
        raise FarmError("task must be a JSON object")
    task_catalog = task.get("crop_catalog")
    crops = load_crops(crop_file) if task_catalog is None else _crops_from_mapping(task_catalog, "task crop_catalog")
    try:
        initial_money = _nonnegative_int(task["initial_money"], "initial_money")
        farmland = _positive_int(task["farmland_count"], "farmland_count")
        duration = _positive_int(task["duration_days"], "duration_days")
        available = task["available_crops"]
    except KeyError as exc:
        raise FarmError(f"task is missing field: {exc.args[0]}") from exc
    if not isinstance(available, list) or not available or not all(isinstance(item, str) for item in available):
        raise FarmError("available_crops must be a non-empty string array")
    if not set(available).issubset(crops):
        raise FarmError("available_crops contains an unknown crop")
    return _greedy_reference_profit(initial_money, farmland, duration, crops, available)


def save_json(value: Any, path: str | Path) -> None:
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _entries(action: Any, field: str, day: int) -> Iterable[tuple[str, int]]:
    if action is None:
        return ()
    if not isinstance(action, list):
        raise FarmError(f"第 {day} 天的 {field} 必须是数组")
    result: list[tuple[str, int]] = []
    for entry in action:
        if not isinstance(entry, dict) or set(entry) != {"crop", "quantity"}:
            raise FarmError(f"第 {day} 天的 {field} 项必须包含 crop 和 quantity")
        crop_id = entry["crop"]
        quantity = _positive_int(entry["quantity"], f"第 {day} 天 {field} 数量")
        if not isinstance(crop_id, str):
            raise FarmError(f"第 {day} 天的作物 ID 必须是字符串")
        result.append((crop_id, quantity))
    return result


def simulate_plan(
    task: dict[str, Any],
    plan: dict[str, Any],
    crop_file: str | Path = DEFAULT_CROP_FILE,
) -> dict[str, Any]:
    """Simulate a plan and return profit, balances, and per-day event log."""
    if not isinstance(task, dict) or not isinstance(plan, dict):
        raise FarmError("task 和 plan 必须是 JSON 对象")
    # Generated tasks carry a crop snapshot so simulation is reproducible even
    # when the global crop data file changes. Older hand-written tasks may omit
    # it, in which case retain the file-based behavior.
    task_catalog = task.get("crop_catalog")
    crops = load_crops(crop_file) if task_catalog is None else _crops_from_mapping(task_catalog, "task crop_catalog")
    try:
        initial_money = _nonnegative_int(task["initial_money"], "initial_money")
        farmland = _positive_int(task["farmland_count"], "farmland_count")
        duration = _positive_int(task["duration_days"], "duration_days")
        available_values = task["available_crops"]
    except KeyError as exc:
        raise FarmError(f"任务缺少字段: {exc.args[0]}") from exc
    if not isinstance(available_values, list) or not available_values or not all(isinstance(item, str) for item in available_values):
        raise FarmError("available_crops 必须是非空字符串数组")
    available = set(available_values)
    if not available or not available.issubset(crops):
        raise FarmError("任务包含未知或空的 available_crops")
    if plan.get("task_id") not in (None, task.get("task_id")):
        raise FarmError("plan.task_id 与 task.task_id 不一致")
    days = plan.get("days", [])
    if not isinstance(days, list):
        raise FarmError("plan.days 必须是数组")

    money = initial_money
    seed_inventory: dict[str, int] = {}
    fruit_inventory: dict[str, int] = {}
    growing: list[dict[str, Any]] = []
    log: list[dict[str, Any]] = []
    seen_days: set[int] = set()

    normalized_days: list[tuple[int, dict[str, Any]]] = []
    for day_record in days:
        if not isinstance(day_record, dict) or "day" not in day_record:
            raise FarmError("每个 days 项必须包含 day")
        normalized_days.append((_positive_int(day_record["day"], "day"), day_record))
    for day, day_record in sorted(normalized_days, key=lambda item: item[0]):
        if day > duration or day in seen_days:
            raise FarmError(f"day 必须在 1..{duration} 且不可重复: {day}")
        seen_days.add(day)
        events: dict[str, Any] = {"day": day, "bought": [], "planted": [], "harvested": [], "sold": [], "money": money}

        for crop_id, quantity in _entries(day_record.get("buy_seeds", []), "buy_seeds", day):
            if crop_id not in available:
                raise FarmError(f"第 {day} 天不可购买作物: {crop_id}")
            cost = crops[crop_id].seed_price * quantity
            if cost > money:
                raise FarmError(f"第 {day} 天资金不足，需要 {cost}，仅剩 {money}")
            money -= cost
            seed_inventory[crop_id] = seed_inventory.get(crop_id, 0) + quantity
            events["bought"].append({"crop": crop_id, "quantity": quantity, "cost": cost})

        for crop_id, quantity in _entries(day_record.get("plant", []), "plant", day):
            if crop_id not in available:
                raise FarmError(f"第 {day} 天不可种植作物: {crop_id}")
            if seed_inventory.get(crop_id, 0) < quantity:
                raise FarmError(f"第 {day} 天 {crop_id} 种子不足")
            if len(growing) + quantity > farmland:
                raise FarmError(f"第 {day} 天农田不足，需要 {quantity}，可用 {farmland - len(growing)}")
            seed_inventory[crop_id] -= quantity
            growing.append({"crop": crop_id, "quantity": quantity, "harvest_day": day + crops[crop_id].maturity_days})
            events["planted"].append({"crop": crop_id, "quantity": quantity})

        for crop_id, quantity in _entries(day_record.get("harvest", []), "harvest", day):
            batches = [batch for batch in growing if batch["crop"] == crop_id and batch["harvest_day"] <= day]
            available_plants = sum(batch["quantity"] for batch in batches)
            if available_plants < quantity:
                raise FarmError(f"第 {day} 天 {crop_id} 尚未有足够成熟作物")
            remaining = quantity
            for batch in list(batches):
                taken = min(remaining, batch["quantity"])
                batch["quantity"] -= taken
                remaining -= taken
                if batch["quantity"] == 0:
                    growing.remove(batch)
                if remaining == 0:
                    break
            fruits = quantity * crops[crop_id].yield_per_harvest
            fruit_inventory[crop_id] = fruit_inventory.get(crop_id, 0) + fruits
            events["harvested"].append({"crop": crop_id, "quantity": quantity, "fruits": fruits})

        for crop_id, quantity in _entries(day_record.get("sell", []), "sell", day):
            if crop_id not in available:
                raise FarmError(f"第 {day} 天不可出售作物: {crop_id}")
            if fruit_inventory.get(crop_id, 0) < quantity:
                raise FarmError(f"第 {day} 天 {crop_id} 果实不足")
            revenue = quantity * crops[crop_id].fruit_price
            fruit_inventory[crop_id] -= quantity
            money += revenue
            events["sold"].append({"crop": crop_id, "quantity": quantity, "revenue": revenue})
        events["money"] = money
        log.append(events)

    # Unsold fruit deliberately has no value; profit is cash change only.
    reference_profit = task.get("reference_profit")
    if reference_profit is None:
        reference_profit = calculate_reference_profit(task, crop_file)
    if not isinstance(reference_profit, (int, float)) or isinstance(reference_profit, bool) or reference_profit < 0:
        raise FarmError("reference_profit must be a non-negative number")
    profit = money - initial_money
    score = (profit / reference_profit) if reference_profit else (1.0 if profit == 0 else None)
    return {
        "task_id": task.get("task_id"),
        "initial_money": initial_money,
        "final_money": money,
        "profit": profit,
        "reference_profit": reference_profit,
        # ``relative_score`` is the shared game-runner contract. ``score`` is
        # retained for compatibility with older farm-only result readers.
        "relative_score": score,
        "score": score,
        "remaining_seeds": {k: v for k, v in seed_inventory.items() if v},
        "unsold_fruits": {k: v for k, v in fruit_inventory.items() if v},
        "growing_crops": growing,
        "daily_log": log,
    }


def _command_generate(args: argparse.Namespace) -> None:
    task = generate_task(args.difficulty, args.crop_file, args.seed)
    save_json(task, args.output)
    print(json.dumps(task, ensure_ascii=False, indent=2))


def _command_simulate(args: argparse.Namespace) -> None:
    task = json.loads(Path(args.task).read_text(encoding="utf-8"))
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    result = simulate_plan(task, plan, args.crop_file)
    if args.output:
        save_json(result, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="文本农场运营规划任务生成与模拟器")
    parser.add_argument("--crop-file", default=str(DEFAULT_CROP_FILE), help="作物基础信息 JSON")
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate", help="生成测试任务")
    gen.add_argument("difficulty", type=int)
    gen.add_argument("--seed", type=int)
    gen.add_argument("-o", "--output", default="task.json")
    gen.set_defaults(func=_command_generate)
    sim = sub.add_parser("simulate", help="模拟 AI 给出的种植计划")
    sim.add_argument("task")
    sim.add_argument("plan")
    sim.add_argument("-o", "--output")
    sim.set_defaults(func=_command_simulate)
    args = parser.parse_args()
    try:
        args.func(args)
    except (FarmError, OSError, json.JSONDecodeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
