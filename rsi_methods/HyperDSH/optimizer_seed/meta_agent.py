"""Fixed HyperDSH optimizer used to evolve only the Harness candidate."""

from agent.base_agent import AgentSystem
from agent.llm_withtools import chat_with_agent


class MetaAgent(AgentSystem):
    """Translate benchmark evidence into one general Harness improvement."""

    def forward(self, repo_path, eval_path, iterations_left=None):
        instruction = f"""Improve the evolvable DeepSeek Harness under
`{repo_path}/harness` using the evaluation evidence in `{eval_path}`.

Read `evidence_manifest.json`, `result.json`, every relevant case, and the
recorded interaction trajectory before editing. Identify a recurring failure
that is supported by concrete observations. Do not optimize for a case id,
seed, exact expected answer, or one stochastic trace.

You may change only these effective Harness surfaces:
- `harness/profile/cordis.patch.yml` for DeepSeek Harness profile behavior;
- `harness/prompt.md` for the task persona and action policy;
- `harness/tools.json` for the enabled tool combination;
- Markdown files below `harness/skills/` for reusable skills.

The supported tool ids are documented in `harness/TOOLS.md`. Keep the profile
manifest on the base + headless bundles. The task runtime is read-only, so do
not design a strategy that depends on modifying the benchmark workspace.
Return task answers in exactly the JSON format requested by each task.

Do not edit the optimizer, Python bridge, benchmark package, Dockerfiles,
requirements, recorded evidence, or files outside `harness/`. Prefer one small,
generalizable change. Inspect the current files after editing and keep JSON
valid. Future refinement calls after this one: {iterations_left}.

You must make the edit yourself with the available tools."""
        return chat_with_agent(
            instruction,
            model=self.model,
            msg_history=[],
            logging=self.log,
            tools_available="all",
            reasoning_effort=self.reasoning_effort,
            backend=self.backend,
            api_mode=self.api_mode,
            api_key_env=self.api_key_env,
            base_url_env=self.base_url_env,
        )
