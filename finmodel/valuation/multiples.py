import math

def enterprise_to_equity(enterprise_value, debt=0, cash=0, minority_interest=0, preferred_equity=0):
    values = (enterprise_value, debt, cash, minority_interest, preferred_equity)
    if not all(math.isfinite(v) for v in values) or min(values[1:]) < 0:
        raise ValueError("Finite values and nonnegative capital bridge components required")
    return enterprise_value - debt - minority_interest - preferred_equity + cash

def implied_multiple(enterprise_value, metric):
    if not math.isfinite(enterprise_value) or not math.isfinite(metric) or metric <= 0:
        raise ValueError("Multiples require finite EV and a positive denominator")
    return enterprise_value / metric
