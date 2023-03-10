from copy import copy
from typing import Optional

from orderbook.models import Order, LimitOrder


class OrderIdConvertor:
    def __init__(self):
        self.external_to_internal_lookup = dict()
        self.counter = 0

    def add_internal_id_to_order_and_track(self, order: LimitOrder) -> LimitOrder:
        self.counter += 1
        new_order = copy(order)
        new_order.internal_id = self.counter
        if order.is_external:
            self.external_to_internal_lookup[order.external_id] = self.counter
        return new_order

    def get_internal_order_id(self, order: Order) -> Optional[int]:
        if not order.is_external:
            return order.internal_id
        elif order.is_external:
            try:
                return self.external_to_internal_lookup[order.external_id]
            except KeyError:
                return None

    def remove_external_order_id(self, external_id: int) -> None:
        try:
            del self.external_to_internal_lookup[external_id]
        except KeyError:
            pass  # If order_id is not present, we ignore it

    def reset(self):
        self.external_to_internal_lookup = dict()
        self.counter = 0
