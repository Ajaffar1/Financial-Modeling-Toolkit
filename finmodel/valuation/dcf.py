from .terminal_value import terminal_value

def discounted_cash_flows(flows, wacc, growth):
    flows = tuple(flows)
    if not flows:
        raise ValueError("Forecast at least one period")
    terminal = terminal_value(flows[-1], wacc, growth) / (1 + wacc) ** len(flows)
    return sum(cf / (1 + wacc) ** t for t, cf in enumerate(flows, 1)) + terminal, terminal
