"""Robust multi-objective joint MA-RIS design under imperfect eavesdropper CSI (RMO branch).

This package is additive: it reuses the validated P0-P4 physical layer unchanged and adds
    encoding.py    joint (T, theta, w) real-valued encoding
    scenarios.py   nested DEV/TEST uncertainty banks with common random numbers
    evaluator.py   direct joint robust evaluation (no inner AO)
    objectives.py  F_mean / LCVaR / Q05 / SOP
    shade.py       SHADE, L-SHADE, L-SHADE+Normal
    moea.py        NSGA-II, MOEA/D-DE, GDE3, SHADE-MOEA, SA-RMOEA
    indicators.py  HV, IGD+, additive epsilon
See reports/RMO_research_plan.md.
"""

__all__ = ["encoding", "scenarios", "evaluator", "objectives", "shade", "moea", "indicators"]
