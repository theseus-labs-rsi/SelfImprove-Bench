# Linux environment and offline checks

The reference check environment is Linux, Python **3.13.15**, git and bubblewrap.
The container definition is [.github/ci/Dockerfile](../.github/ci/Dockerfile).
Direct Python requirements and the two check tools are pinned in
`requirements.txt` / `requirements-dev.txt`; NexAU is the bundled wheel.
The check records the interpreter, platform, direct dependency versions and a
full `pip freeze`. Transitive dependencies and Debian packages are not a
hash-locked distribution: retain the image ID and check artifacts when comparing
machines or reproducing a published run.

## Recommended: use the same network-isolated container as CI

Run from the repository root, with Docker available:

```bash
# Explicit online preparation: install packages and fetch public tokenizer data.
# This stage needs no model credentials and performs no inference.
docker build -f .github/ci/Dockerfile -t rsi-bench-check .

mkdir -p outputs/checks
docker run --rm --network none \
  --mount type=bind,src="$PWD",dst=/repo,readonly \
  --mount type=bind,src="$PWD/outputs/checks",dst=/checks \
  rsi-bench-check python -m scripts.check --output-dir /checks
```

The checkout is read-only, results are written to `/checks`, and the test
container has no network or forwarded model credentials. No Docker daemon is
mounted into the test container: the tests of Docker adapters use mocks.
The image build is a preparation step; the checks never install packages or
download assets implicitly. A build/download error is not a passing offline run.

### Existing-cache mode when the asset endpoint is unavailable

If the final tokenizer download fails (including TLS/network errors), keep TLS
verification enabled. Build only the named `dependencies` stage and supply an
existing verified cache instead. The following uses artifact/cache directories
outside the checkout; replace the two absolute paths before running:

```bash
docker build --target dependencies -f .github/ci/Dockerfile -t rsi-bench-check-dependencies .
mkdir -p /absolute/path/to/rsi-check-artifacts
docker run --rm --network none \
  --mount type=bind,src="$PWD",dst=/repo,readonly \
  --mount type=bind,src=/absolute/path/to/existing/tiktoken_cache,dst=/source-cache,readonly \
  --mount type=bind,src=/absolute/path/to/rsi-check-artifacts,dst=/checks \
  -e TIKTOKEN_CACHE_DIR=/checks/tokenizer_cache \
  rsi-bench-check-dependencies python -m scripts.test_assets --source-dir /source-cache
docker run --rm --network none \
  --mount type=bind,src="$PWD",dst=/repo,readonly \
  --mount type=bind,src=/absolute/path/to/rsi-check-artifacts,dst=/checks \
  -e TIKTOKEN_CACHE_DIR=/checks/tokenizer_cache \
  rsi-bench-check-dependencies python -m scripts.check --output-dir /checks
```

The copy verifies the same SHA-256 and makes no network request. No asset is
committed or copied into the repository. This verifies the dependency-stage
image with a supplied cache; it does not certify that the default online image
build/download worked on that host. CI builds the default final `checks` stage.

## Native Linux preparation

Use a dedicated Python 3.13 environment; Python 3.13.15 is the tested reference.
Install git and bubblewrap with your Linux package manager, then:

```bash
python3.13 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-dev.txt

# Explicit network access for one public tokenizer data file, not a model API.
python -m scripts.test_assets --download
python -m scripts.check
```

`TIKTOKEN_CACHE_DIR` selects the cache; otherwise tooling uses
`$HOME/.cache/rsi-bench/tiktoken` (or the matching `XDG_CACHE_HOME` path), outside
the checkout. To prepare an air-gapped machine from an
existing tiktoken cache instead of downloading:

```bash
python -m scripts.test_assets --source-dir /path/to/existing/tiktoken_cache
python -m scripts.check --preflight-only
python -m scripts.check
```

The resource is `cl100k_base.tiktoken` from the public tiktoken asset endpoint.
Both copy and download verify SHA-256
`223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7`
before atomically installing it under tiktoken's URL-derived cache filename.
The default resource command only verifies; missing or corrupt data fails.
This avoids ACE initialization unexpectedly trying to download a tokenizer
during an offline test. Preparing this data does not require an API key.

## What the single check runs

`python -m scripts.check` performs:

1. A Linux/Python 3.13, exact direct dependency, git, tokenizer integrity and
   bubblewrap namespace preflight; failures give an actionable message before tests.
2. `pip check` and an environment snapshot.
3. Focused Ruff correctness rules `E9,F63,F7,F82` on first-party Python files.
   No formatting rewrite, auto-fix or vendor linting is performed.
4. A narrow mypy fixture that reads the real `TaskEvaluation` contract and
   checks an optional-score consumer. Runtime/provider imports are not followed;
   this is not a claim of full-project type safety.
5. The first-party pytest suite: `tests/`, Farm/Combat planner regressions,
   GameWorld retry tests and the HyperAgents wrapper tests. Newly added files
   under `tests/` are included automatically. Vendored upstream suites are excluded.

Subprocesses receive a small environment without model credentials or proxy
settings, disable bytecode and pytest plugin/cache discovery, and inherit a
Python audit guard that rejects DNS and Internet socket connections. This guard
is a defense against accidental Python requests, **not an OS sandbox**; a
subprocess can replace its Python import path. The container command and CI
therefore also use Docker `--network none` to cover all descendant processes.

Logs and commands are saved in `outputs/checks/`: `checks.json`, `pip-check.log`,
`pip-freeze.log`, `ruff.log`, `score-contract-types.log`, `pytest.log`, `pytest.xml`.
An unsuccessful stage returns nonzero and retains its log. `--skip-tests` runs
preflight plus static checks; `--output-dir` changes the artifact destination.
Read the pytest skip summary rather than treating all collected tests as passed.
JUnit can include unittest subtests; report item, subtest and skip counts separately.

## Conditional integration checks and full experiments

Linux containers/hosts can deny unprivileged user namespaces even when bwrap is
installed. The preflight reports the actual probe; namespace integration tests
then skip with their reason. For a gate that must verify isolation, use
`python -m scripts.check --require-namespaces`: it fails preflight if the probe
cannot run. Do not disable AHE's sandbox or turn a skipped probe into a pass.

Offline tests construct actual runtime objects where practical, and use
model-free fakes for model calls, browser launch and Docker clients. They do not
prove three real methods train/resume correctly end to end. Full experiments
add separate requirements: configured model endpoints/credentials, Docker for
HyperAgents, working namespaces for AHE, and an installed GameWorld runtime plus
Chromium for visual games (see the root setup instructions). Renderer settings,
browser version, method configuration and environment snapshots should accompany
those experiment artifacts. The offline command does not contact or validate a
model endpoint, start a real browser game, or launch a method's Docker workload.

Raw `TaskEvaluation.score=None` means the task score is unavailable; an observed
zero remains `0.0`. Summary policies can count failed/unscored attempts as zero
while preserving the raw evidence. Unknown token/time fees remain `None` and
must not be presented as free calls. The regression suite checks these contracts;
the score-consumer mypy fixture checks only the optional-score boundary, not all
business logic or the fidelity of real provider usage reports.

The GitHub workflow runs this same command after its explicit preparation stage
and uploads the logs on success or failure. It requests read-only repository
permissions and does not pass model secrets to PR jobs.
