"""Reservation integral shared by candidate features and accounting."""
import numpy as np
from slice_alloc import patrol

def marginal_cost(ledger, plan):
    start, end = plan.request.start, plan.request.drain
    points = {start, end}
    for existing in ledger.plans + [plan]:
        points.update(t for a, b, _ in existing.segments for t in (a, b) if start < t < end)
        if existing.lease:
            points.update(t for t in existing.lease[-2:] if start < t < end)
    integral = np.zeros(len(patrol.RESOURCES))
    price = 0.
    points = sorted(points)
    for a, b in zip(points, points[1:]):
        usage = ledger.usage((a + b) / 2)
        addition = (ledger.usage((a + b) / 2, plan) - usage) / ledger.capacity
        integral += (b - a) * addition
        price += (b - a) * float(np.dot(addition, 1 + 4 * (usage / ledger.capacity) ** 2))
    return integral, price / (end - start)
