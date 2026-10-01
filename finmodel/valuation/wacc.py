def wacc(equity, debt, cost_of_equity, cost_of_debt, tax_rate):
    if equity < 0 or debt < 0 or equity + debt <= 0 or not 0 <= tax_rate <= 1:
        raise ValueError("Invalid capital weights or tax rate")
    return (equity * cost_of_equity + debt * cost_of_debt * (1 - tax_rate)) / (equity + debt)
