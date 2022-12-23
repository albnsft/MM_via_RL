from orderbook.models import Orderbook


class OrderDistributor:
    def __init__(self, volume: int = 100):
        self.volume = volume

    @property
    def limit_orders(self):
        """
        A typical action-space for market making, tuple being the value of (Ask (θa), Bid (θb))
        Smaller values of θa,b lead to quotes that are closer to the top of the book, while larger
        numbers cause the quotes to be deeper in the book.
        actions 0 through 8 cause limit orders to be placed at fixed distances relative to a reference price, Ref(ti),
        chosen here to be a measure of the true market price: the mid-price.
        """
        return {0: (1, 1), 1: (2, 2), 2: (3, 3), 3: (4, 4), 4: (5, 5), 5: (1, 3), 6: (3, 1), 7: (2, 5),
                        8: (5, 2)}

    @staticmethod
    def distance_price(tetha_sell: int, tetha_buy: int, spread: float) -> tuple[float, float]:
        distance_sell = tetha_sell * spread/2
        distance_buy = tetha_buy * spread/2
        return distance_sell, distance_buy

    def pricing_strat(self, tetha_sell: int, tetha_buy: int, spread: float, mid_price: float) -> tuple[float, float]:
        distance_sell, distance_buy = self.distance_price(tetha_sell, tetha_buy, spread)
        price_sell = mid_price + distance_sell
        price_buy = mid_price - distance_buy
        return price_sell, price_buy

    def convert_action(self, action: int = None, orderbook: Orderbook = None) -> tuple[int, int, dict[str, float]]:
        assert action in list(range(10))
        tetha_sell, tetha_buy = self.limit_orders.get(action)
        spread = orderbook.spread
        midprice = orderbook.midprice
        price_sell, price_buy = self.pricing_strat(tetha_sell, tetha_buy, spread, midprice)
        return tetha_sell, tetha_buy, {"buy": price_buy, "sell": price_sell}

