"""Back-compat shim.

The in-process bus moved to ``bubbles.bus.local.LocalBus`` when the bus
transport became pluggable (Phase 2). ``MessageBus`` here is kept as an alias
so existing imports (``from bubbles.bus.queue import MessageBus``) and the
``bus=`` injection sites keep working. New code should depend on the
``bubbles.bus.base.MessageBus`` ABC and construct instances via
``bubbles.bus.factory.make_bus``.
"""

from bubbles.bus.local import LocalBus as MessageBus

__all__ = ["MessageBus"]
