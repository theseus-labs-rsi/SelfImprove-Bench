# 农场运营规划任务 Prompt

你是一名农场经营规划 AI。请只依据输入任务中的 `available_crops` 和作物基础信息制定计划，在任务持续天数内最大化最终现金盈利。

## 输入模板

```json
{
  "task_id": "farm-example",
  "difficulty": 2,
  "initial_money": 200,
  "farmland_count": 12,
  "duration_days": 12,
  "available_crops": ["wheat", "corn"],
  "crop_catalog": {
    "wheat": {"id": "wheat", "name": "小麦", "seed_price": 20, "maturity_days": 2, "yield_per_harvest": 3, "fruit_price": 12},
    "corn": {"id": "corn", "name": "玉米", "seed_price": 35, "maturity_days": 3, "yield_per_harvest": 4, "fruit_price": 16}
  }
}
```

规则：第 `day` 天购买的种子才可在当天或之后种植；收获日为种植日加 `maturity_days`；每天操作顺序为购买、种植、收获、出售；农田在收获前持续占用；未出售果实和剩余种子不计入盈利；不能透支资金、种植未知作物或出售未收获果实。

任务 JSON 中只包含制定计划所需的规则、资金、农田和作物信息；请最大化实际模拟利润。

## 输出模板

只输出合法 JSON，不要添加 Markdown 围栏或解释文字：

```json
{
  "task_id": "farm-example",
  "days": [
    {
      "day": 1,
      "buy_seeds": [{"crop": "wheat", "quantity": 4}],
      "plant": [{"crop": "wheat", "quantity": 4}],
      "harvest": [],
      "sell": []
    },
    {
      "day": 3,
      "buy_seeds": [],
      "plant": [],
      "harvest": [{"crop": "wheat", "quantity": 4}],
      "sell": [{"crop": "wheat", "quantity": 12}]
    }
  ]
}
```

可以省略没有操作的日期；每个日期最多出现一次。所有数量必须是正整数。目标是最大化模拟器返回的 `profit`。
