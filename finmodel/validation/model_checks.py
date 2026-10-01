from dataclasses import dataclass
@dataclass(frozen=True)
class Finding:
    code: str
    severity: str
    message: str
    year: int | None = None

def audit_model(model, tolerance=1e-6):
    findings = []
    previous_cash = model.opening.cash
    for period in model.periods:
        b, c = period.balance_sheet, period.cash_flow
        for code, error in [("BALANCE_SHEET", b.balance_error),
                            ("CASH_RECONCILIATION", b.cash - previous_cash - c.net_change)]:
            if abs(error) > tolerance:
                findings.append(Finding(code, "error", f"Unreconciled amount: {error:,.6f}", period.year))
        for code, value in [("NEGATIVE_CASH", b.cash), ("NEGATIVE_DEBT", b.debt), ("NEGATIVE_PPE", b.ppe)]:
            if value < -tolerance:
                findings.append(Finding(code, "error", f"Closing value: {value:,.2f}", period.year))
        previous_cash = b.cash
    ev, terminal = model._valuation()
    if ev <= 0:
        findings.append(Finding("NONPOSITIVE_EV", "warning", "Enterprise value is nonpositive"))
    elif terminal / ev > 0.75:
        findings.append(Finding("TERMINAL_CONCENTRATION", "warning", f"Terminal value is {terminal / ev:.1%} of EV"))
    return findings
