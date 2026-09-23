"""
ACE (Agent-Curator-Environment) System

A playbook-based learning framework with three main components:
- Generator: Produces answers using playbook knowledge
- Reflector: Analyzes outputs and provides feedback
- Curator: Updates the playbook based on feedback

Usage:
    from ace import ACE
    
    ace_system = ACE(
        generator_client=client,
        reflector_client=client,
        curator_client=client,
        generator_model="model-name",
        reflector_model="model-name",
        curator_model="model-name"
    )
    
    # Offline adaptation
    results = ace_system.run(
        mode='offline',
        train_samples=train_data,
        val_samples=val_data,
        test_samples=test_data,  # Optional
        data_processor=processor,
        config=config
    )

    # Online adaptation
    results = ace_system.run(
        mode='online',
        test_samples=test_data,
        data_processor=processor,
        config=config
    )

    # Evaluation only
    results = ace_system.run(
        mode='eval_only',
        test_samples=test_data,
        data_processor=processor,
        config=config
    )
"""

__all__ = ['ACE', 'ACEBatch', 'Generator', 'Reflector', 'Curator', 'BulletpointAnalyzer']

__version__ = "1.0.0"


def __getattr__(name):
    """Preserve the public API without importing every optional component."""
    if name == "ACE":
        from .ace import ACE
        value = ACE
    elif name == "ACEBatch":
        from .ace_batch import ACEBatch
        value = ACEBatch
    elif name in {"Generator", "Reflector", "Curator", "BulletpointAnalyzer"}:
        from importlib import import_module
        core = import_module(f"{__name__}.core")
        value = getattr(core, name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    globals()[name] = value
    return value
