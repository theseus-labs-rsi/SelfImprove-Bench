# code_agent_simple

A minimal NexAU code agent that only registers `run_shell_command`; see
`code_agent.yaml` for the configuration.

## Local testing

```bash
export LLM_MODEL=... LLM_BASE_URL=... LLM_API_KEY=...
export SANDBOX_WORK_DIR=/tmp/code_agent_work   # recommended
python3 start.py
```

Or via the NexAU CLI (this directory must be the working directory and `nexau`
must be installed):

```bash
nexau run code_agent
```
