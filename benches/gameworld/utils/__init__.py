"""Runtime helpers for GameWorld browser snapshots and durable step recovery.

Astray and Minecraft keep separate capture/restore modules; shared state and
browser instrumentation live alongside them. Import the required module from
this package rather than loading helpers through the experiment's tools folder.
"""
