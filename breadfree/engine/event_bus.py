"""
BreadFree EventBus

A lightweight publish-subscribe event system inspired by vnpy's EventEngine.

- Backtest mode: synchronous dispatch (events processed in order)
- Live mode: can be extended to async dispatch (asyncio + queue)

Usage:
    bus = EventBus()
    bus.subscribe(EventType.ORDER, my_handler)
    bus.publish(Event(EventType.ORDER, order_data))
"""

from collections import defaultdict
from typing import Callable, List, Dict
from .models import Event, EventType
from ..utils.logger import get_logger

logger = get_logger(__name__)


class EventBus:
    """
    Synchronous event bus for decoupling modules.

    In backtest mode, events are dispatched immediately (synchronous).
    For live trading, this can be extended with asyncio queues.
    """

    def __init__(self):
        self._handlers: Dict[EventType, List[Callable]] = defaultdict(list)
        self._general_handlers: List[Callable] = []  # Handlers for ALL events

    def subscribe(self, event_type: EventType, handler: Callable):
        """
        Subscribe a handler to a specific event type.

        :param event_type: The type of event to listen for
        :param handler: Callable that accepts an Event object
        """
        if handler not in self._handlers[event_type]:
            self._handlers[event_type].append(handler)
            logger.debug(f"EventBus: subscribed {handler.__name__} to {event_type.value}")

    def subscribe_all(self, handler: Callable):
        """Subscribe a handler to ALL event types (useful for logging/audit)."""
        if handler not in self._general_handlers:
            self._general_handlers.append(handler)

    def unsubscribe(self, event_type: EventType, handler: Callable):
        """Remove a handler from a specific event type."""
        handlers = self._handlers.get(event_type, [])
        if handler in handlers:
            handlers.remove(handler)

    def publish(self, event: Event):
        """
        Publish an event to all subscribed handlers (synchronous).

        :param event: The event to dispatch
        """
        # Dispatch to type-specific handlers
        for handler in self._handlers.get(event.event_type, []):
            try:
                handler(event)
            except Exception as e:
                logger.error(
                    f"EventBus: handler {handler.__name__} failed on "
                    f"{event.event_type.value}: {e}"
                )

        # Dispatch to general handlers
        for handler in self._general_handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(
                    f"EventBus: general handler {handler.__name__} failed: {e}"
                )

    def clear(self):
        """Remove all handlers."""
        self._handlers.clear()
        self._general_handlers.clear()
