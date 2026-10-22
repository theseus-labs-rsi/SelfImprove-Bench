"""HyperAgents infrastructure failures must not become benchmark scores."""

from benchmark.methods import MethodRuntimePolicy


class HyperAgentsRuntimePolicy(MethodRuntimePolicy):
    def is_fatal_error(self, value):
        message = str(value or "").lower()
        return any(marker in message for marker in (
            "hyperagents network preflight failed",
            "docker image build failed",
            "docker daemon is unavailable",
            "docker container creation failed",
        ))
