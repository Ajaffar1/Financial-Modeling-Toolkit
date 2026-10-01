from dataclasses import asdict

def assumption_changes(base, scenario):
    """Return explicit assumption differences for scenario review."""
    left, right = asdict(base.assumptions), asdict(scenario.assumptions)
    return {key: (left[key], right[key]) for key in left if left[key] != right[key]}
