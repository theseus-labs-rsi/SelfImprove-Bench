# Farm operation planning task prompt

You are a farm operation planning AI. Base your plan only on the `available_crops`
and crop base information in the input task, and maximize the final cash profit within
the task's duration.

## Input template

```json
{
  "task_id": "farm-example",
  "difficulty": 2,
  "initial_money": 200,
  "farmland_count": 12,
  "duration_days": 12,
  "available_crops": ["wheat", "corn"],
  "crop_catalog": {
    "wheat": {"id": "wheat", "name": "Wheat", "seed_price": 20, "maturity_days": 2, "yield_per_harvest": 3, "fruit_price": 12},
    "corn": {"id": "corn", "name": "Corn", "seed_price": 35, "maturity_days": 3, "yield_per_harvest": 4, "fruit_price": 16}
  }
}
```

Rules: seeds bought on day `day` can only be planted on that day or later; the harvest day is the plant day plus `maturity_days`; farmland stays occupied until harvest; unsold fruit and remaining seeds do not count toward profit; you cannot overdraw funds, plant unknown crops, or sell unharvested fruit.

Within each day, operations execute in the order their keys appear in your JSON object.
There is no fixed operation order; put each prerequisite before the operation that needs it.

The task JSON contains only the rules, funds, farmland, and crop information needed to build a plan; maximize the actual simulated profit.

## Output template

Output valid JSON only, with no Markdown fences or explanatory text:

```json
{
  "task_id": "farm-example",
  "days": [
    {
      "day": 1,
      "harvest": [],
      "sell": [],
      "buy_seeds": [{"crop": "wheat", "quantity": 4}],
      "plant": [{"crop": "wheat", "quantity": 4}]
    },
    {
      "day": 3,
      "harvest": [{"crop": "wheat", "quantity": 4}],
      "sell": [{"crop": "wheat", "quantity": 12}],
      "buy_seeds": [],
      "plant": []
    }
  ]
}
```

You may omit days with no operations; each day may appear at most once. All quantities
must be positive integers. The goal is to maximize the `profit` returned by the simulator.
