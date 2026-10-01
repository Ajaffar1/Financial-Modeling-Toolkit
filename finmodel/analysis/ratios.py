def statement_ratios(period):
    """Undefined coverage/leverage metrics are None rather than misleading infinities."""
    i, b = period.income_statement, period.balance_sheet
    divide = lambda n, d: n / d if d > 0 else None
    return {"ebitda_margin": divide(i.ebitda, i.revenue), "net_margin": divide(i.net_income, i.revenue),
            "debt_to_ebitda": divide(b.debt, i.ebitda), "net_debt_to_ebitda": divide(b.debt - b.cash, i.ebitda),
            "interest_coverage": divide(i.ebit, i.interest)}
