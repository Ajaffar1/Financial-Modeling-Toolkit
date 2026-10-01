from ..analysis.ratios import statement_ratios
from .model_checks import Finding

def check_covenants(model, *, maximum_debt_to_ebitda=None, minimum_interest_coverage=None, minimum_cash=0):
    """Evaluate reporting-date thresholds; not a legal credit agreement parser."""
    import math
    thresholds = (maximum_debt_to_ebitda, minimum_interest_coverage, minimum_cash)
    if any(v is not None and (not math.isfinite(v) or v < 0) for v in thresholds):
        raise ValueError("Covenant thresholds must be finite and nonnegative")
    if not model.periods:
        raise ValueError("Forecast the model before checking covenants")
    findings = []
    for p in model.periods:
        ratios = statement_ratios(p)
        if maximum_debt_to_ebitda is not None:
            ratio = ratios['debt_to_ebitda']
            if ratio is None or ratio > maximum_debt_to_ebitda:
                findings.append(Finding('LEVERAGE_COVENANT','error','Leverage exceeds threshold or EBITDA is nonpositive',p.year))
        if minimum_interest_coverage is not None and p.income_statement.interest > 0:
            ratio = ratios['interest_coverage']
            if ratio is None or ratio < minimum_interest_coverage:
                findings.append(Finding('COVERAGE_COVENANT','error','Interest coverage below threshold',p.year))
        if p.balance_sheet.cash < minimum_cash:
            findings.append(Finding('LIQUIDITY_COVENANT','error','Cash below minimum balance',p.year))
    return findings
