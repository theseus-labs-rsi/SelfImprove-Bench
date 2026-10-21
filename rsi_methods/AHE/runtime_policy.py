"""AHE runtime rules, kept outside the benchmark orchestration."""

from benchmark.methods import MethodRuntimePolicy


def is_fatal_error(value):
    message = str(value or "").lower()
    return any(marker in message for marker in (
        "ahe sandbox", "ahe selection", "ahe candidate generation failed",
        "ahe refine needs a trace", "ahe agent debugger", "budget has been exceeded",
        "context too full", "pending rsi candidate batch", "prompt too long",
    ))


class AHERuntimePolicy(MethodRuntimePolicy):
    reload_holdout_agent = True
    is_fatal_error = staticmethod(is_fatal_error)

    def normalize_result(self, result):
        error = str(result.get("error") or "").lower()
        if result.get("status") != "error" or not any(marker in error for marker in (
            "no response content or tool calls from llm", "empty response from api",
        )):
            return result
        return {**result, "status": "skipped", "relative_score": 0.0,
                "raw_score": 0.0, "success": False,
                "observed_relative_score": result.get("relative_score"),
                "skip_reason": "ahe_empty_response"}

    def selection_error(self, error):
        return f"AHE selection failed: {error}"

    def scheduled_refinements(self, count, adaptor):
        return count // adaptor.refinement_every

    def interrupted_paths(self, adaptor):
        iteration = adaptor._refinement_round + 1
        return [adaptor.run_dir / kind / f"iteration_{iteration:03d}"
                for kind in ("runs", "candidates")]
