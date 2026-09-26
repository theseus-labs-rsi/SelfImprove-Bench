"""
Import built-in task modules to trigger registration.
"""
from herosjourney.core.tasks import additive      # noqa: F401
from herosjourney.core.tasks import compositional  # noqa: F401
from herosjourney.core.tasks import conditional    # noqa: F401
from herosjourney.core.tasks import override       # noqa: F401
from herosjourney.core.tasks import proc_comp      # noqa: F401
from herosjourney.core.tasks import proc_cond      # noqa: F401
from herosjourney.core.tasks import proc_over      # noqa: F401
from herosjourney.core.tasks import proc_add       # noqa: F401
from herosjourney.core.tasks import proc_over_deep # noqa: F401
from herosjourney.core.tasks import proc_add_deep   # noqa: F401
from herosjourney.core.tasks import proc_chain_deep # noqa: F401
from herosjourney.core.tasks import proc_chain_dual # noqa: F401
from herosjourney.core.tasks import proc_chain_nested # noqa: F401
from herosjourney.core.tasks import proc_chain_quint # noqa: F401
