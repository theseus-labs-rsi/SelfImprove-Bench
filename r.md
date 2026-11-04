nohup python -m benchmark ahe_combat.json > ahe_combat.log 2>&1 &
nohup python -m benchmark ahe_farm.json > ahe_farm.log 2>&1 &

nohup python -m benchmark ahe_0.json > ahe_0.log 2>&1 &
nohup python -m benchmark ace_0.json > ace_0.log 2>&1 &
nohup python -m benchmark hyper_0.json > hyper_0.log 2>&1 &

nohup python -m benchmark hyper_farm_cross.json > hyper_farm_cross.log 2>&1 &
nohup python -m benchmark hyper_combat_cross.json > hyper_combat_cross.log 2>&1 &
nohup python -m benchmark hyper_hero_cross.json > hyper_hero_cross.log 2>&1 &
nohup python -m benchmark hyper_mine_cross.json > hyper_mine_cross.log 2>&1 &

nohup python -m benchmark hyper_combat_mix.json > hyper_combat_mix.log 2>&1 &
nohup python -m benchmark hyper_combat_learn.json > hyper_combat_learn.log 2>&1 &
nohup python -m benchmark hyper_combat_repeat.json > hyper_combat_repeat.log 2>&1 &
nohup python -m benchmark hyper_hero_mix.json > hyper_hero_mix.log 2>&1 &
nohup python -m benchmark hyper_hero_learn.json > hyper_hero_learn.log 2>&1 &
nohup python -m benchmark hyper_hero_repeat.json > hyper_hero_repeat.log 2>&1 &

nohup python -m benchmark hyperdsh_combat.json > hyperdsh_combat.log 2>&1 &
nohup python -m benchmark hyperdsh_hero.json > hyperdsh_hero.log 2>&1 &

nohup python -m benchmark hyperdsh_combat_0.json > hyperdsh_combat_0.log 2>&1 &
nohup python -m benchmark hyperdsh_hero_0.json > hyperdsh_hero_0.log 2>&1 &

nohup python -m benchmark hyperdsh_combat_claude.json > hyperdsh_combat_claude.log 2>&1 &

python -m analyze hyperdsh "C:\Users\35810\Desktop\hyperdsh_combat\hyperdsh_combat" --model gpt-5.6-sol  --output "D:\coding\projects\rsi-research\hyperdsh_combat_analysis.json"

python -m analyze hyperdsh "C:\Users\35810\Desktop\hyperdsh_combat\hyperdsh_combat" --model gpt-5.6-sol  --output "D:\coding\projects\rsi-research\hyperdsh_combat_analysis.json"

nohup python -m analyze hyperdsh "/home/bangrui/RSI-Bench-CSD/benchmark_results/hyperdsh_combat1" --model "gpt-5.6-sol"  --output "/home/bangrui/RSI-Bench-CSD/analyze/hyperdsh_combat_ds.json" > analyze1.log 2>&1 &
nohup python -m analyze hyperdsh "/home/bangrui/RSI-Bench-CSD/benchmark_results/hyperdsh_combat_01" --model "gpt-5.6-sol"  --output "/home/bangrui/RSI-Bench-CSD/analyze/hyperdsh_combat_0.json" > analyze2.log 2>&1 &


  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="/home/bangrui/.local/bin:$PATH"

  uv python install 3.13
  cd /home/bangrui/RSI-Bench-CSD/rsi_methods/AHE/agentic-harness-engineering
  uv sync --python 3.13

  然后使用该虚拟环境运行 benchmark：

  source .venv/bin/activate
  cd /home/bangrui/RSI-Bench-CSD
  python -m benchmark ahe_combat.json


1. 数据质量和规模的系统性提升。不同架构、不同参数规模的模型在同一批难样本上呈现高度一致的失败模式，表明瓶颈来自训练数据的共性问题，而非架构差异。因此方法的核心观点是数据scaling而非模型scaling。
(1) 通过CMC（跨模型一致性验证） 将样本自动分为Easy/Medium/Hard三级，Easy 和 Medium 直接获得可靠标注。
(2) 通过Judge-and-Refine流程对难例进行迭代纠正，剩余样本转专家标注，确保标注精度。
(3) DDAS（多样性与难度感知采样）在页面级和元素级同时进行聚类与难度评估，对长尾类别上采样、对高频类别下采样，并对 Medium / Hard 样本加权，提升训练信号密度。

2. 评测协议改进。不同OCR的输出会有主观差异，尤其是复杂布局的分块粒度上，需要保证对不同粒度输出的评测公平性与区分度。
(1) 修正元素匹配偏差，原本的OmniDocBench v1.5 采用固定粒度一对一匹配，导致语义正确但分块粒度不同的预测被严重扣分。v1.6 引入 MGAM（多粒度自适应匹配），在预测侧自适应调整粒度，消除格式偏好带来的评分偏差。
(2) 增加难度分层与 Hard 子集，建立Base/Hard/Full三级评估协议，Base保持与v1.5可比，Hard提供更具区分度的衡量，Full为并集。

3. 各子任务识别能力提升
(1) 文本识别，针对手写体、竖版文言文等难例进行优化；增加对因多栏、插图导致截断的连贯文本的识别能力。
(2) 公式识别，对复杂公式采取分治策略，提升对微积分、指数、分数等结构的识别能力。
(3) 表格识别，增加对半结构化表格、表格中包含图片、多个并列子图等复杂案例的识别能力。

export OPENAI_API_KEY="sk-kDpHJ4uSEDwDOlENF0KQwHhKaTKC5NKbhEfeLLmoh6tecmsX"
export OPENAI_BASE_URL="https://free-router.opendatalab.com/v1"


nohup python -m benchmark configs/main/combat_hyperagents.json > hyperagents_combat.log 2>&1 &
nohup python -m benchmark configs/main/farm_hyperagents.json > hyperagents_farm.log 2>&1 &


nohup python -m benchmark configs/main/farm_ace.json > ace_farm.log 2>&1 &
nohup python -m benchmark configs/main/farm_ahe.json > ahe_farm.log 2>&1 &

nohup python -m benchmark configs/main/combat_ace_g4.json > combat_ace_g4.log 2>&1 &
nohup python -m benchmark configs/main/combat_ahe_g4.json > combat_ahe_g4.log 2>&1 &

nohup python -m benchmark hyperdsh_farm.json > hyperdsh_farm.log 2>&1 &
nohup python -m benchmark hyperdsh_sweepmine.json > hyperdsh_sweepmine.log 2>&1 &
nohup python -m benchmark hyperdsh_herojourney.json > hyperdsh_herojourney.log 2>&1 &

nohup python -m benchmark configs/main/astray_hyperdsh.json > hyperdsh_astray.log 2>&1 &
nohup python -m benchmark configs/main/minecraft_hyperdsh.json > hyperdsh_minecraft.log 2>&1 &

nohup python -m benchmark configs/combat_harness2_d3_5train_1eval.json > harness2_test.log 2>&1 &
nohup python -m benchmark configs/combat_recuris_d3_5train_1eval.json > recuris_test.log 2>&1 &
nohup python -m benchmark configs/combat_rrsi_d3_5train_1eval.json > rrsi_test.log 2>&1 &
nohup python -m benchmark configs/combat_skillevolver_d3_5train_1eval.json > skillevolver_test.log 2>&1 &
nohup python -m benchmark configs/combat_skillopt_d3_5train_1eval.json > skillopt_test.log 2>&1 &