"""Scoped optimizer transport for embedders, including native worker pools."""
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar, copy_context

optimizer_transport = ContextVar("skillopt_optimizer_transport", default=None)


class ContextExecutor(ThreadPoolExecutor):
    def submit(self, fn, /, *args, **kwargs):
        context = copy_context()
        return super().submit(context.run, fn, *args, **kwargs)
