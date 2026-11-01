"""Reproduce design-level counterexamples and contract tests; no RL/ns-3 claims.

Run in sdn: python -B -m slice_alloc.tools.design_checks
"""
import hashlib
import io
import json
from pathlib import Path
import unittest
import numpy as np
from scipy.optimize import linprog

ROOT = Path(__file__).resolve().parents[2]


def expression_diagnostics():
    """Synthetic two-resource ranking checks, not measured network results."""
    def barrier(q, occupancy=(0., 0.)):
        u, q = np.asarray(occupancy), np.asarray(q)
        if np.any(q < 0) or np.any(u < 0) or np.any(u + q > 1):
            raise ValueError("infeasible synthetic occupancy/allocation")
        return float(np.log((1.05-u) / (1.05-u-q)).sum())

    # price vector >= 0, normalized to sum=1. Maximize the margin by which C
    # can beat BOTH alternatives. Negative optimum proves no strict preference.
    result = linprog([0., 0., -1.], A_ub=[[.6, -.4, 1.], [-.4, .6, 1.]],
                     b_ub=[0., 0.], A_eq=[[1., 1., 0.]], b_eq=[1.],
                     bounds=[(0., None), (0., None), (None, None)], method="highs")
    if not result.success:
        raise RuntimeError(result.message)
    q = [np.array(v) for v in ((0., 1.), (1., 0.), (.6, .6))]
    def synthetic_ranking(occupancy, choices, price):
        # Self-contained mathematical illustration, not a current online
        # baseline call. The deployed baseline is in joint.choose.
        scores=[0.]+[100.-.003*float(v.sum())-price*barrier(v,occupancy)
                     for v in choices]
        return int(np.argmax(scores))
    actual_choice = synthetic_ranking((0.,0.),q,.3)
    a, b = np.array([.05, .90]), np.array([.5, .5])
    c, d = np.array([.6, .6]), np.array([0., 1.])
    coefficient = .003
    # Observable resource states differ. These are stipulated rankings, NOT
    # rankings established as optimal by giving a policy hidden future demands.
    u1, u2 = (.02, 0.), (.1, 0.)
    threshold1 = -coefficient * (a.sum() - b.sum()) / (barrier(a,u1) - barrier(b,u1))
    threshold2 = coefficient * (c.sum() - d.sum()) / (barrier(d,u2) - barrier(c,u2))
    checks = [synthetic_ranking(u1,[a,b],threshold1*.5)==1,
              synthetic_ranking(u1,[a,b],threshold1*2)==2,
              synthetic_ranking(u2,[c,d],threshold2*.5)==2,
              synthetic_ranking(u2,[c,d],threshold2*2)==1]
    return {
        "kind": "synthetic mathematical checks, not simulation or learning results",
        "linear_vector_price_best_margin": float(result.x[2]),
        "barrier_penalties": {k: barrier(v) for k, v in zip("ABC", q)},
        "synthetic_barrier_choice": "ABC"[actual_choice - 1] if actual_choice else "reject",
        "fixed_scalar_two_regimes": {
            "cost_coefficient": coefficient,
            "observable_occupancies": [list(u1),list(u2)],
            "synthetic_barrier_threshold_checks": checks,
            "desired_rankings_are_assumptions_not_proven_optima": True,
            "regime1_requires_p_below": threshold1,
            "regime2_requires_p_above": threshold2,
            "intersection_empty": bool(threshold2 >= threshold1),
            "scope": "algebraic limitation for stipulated rankings at distinct observed occupancies; no RL superiority claim",
            "online_caveat": "Different unknown futures alone are not observable states; no online policy can distinguish them."
        },
        "sac_caveat": "The withdrawn proposal used a state-dependent RESOURCE VECTOR, not merely two scalars."
    }


def main():
    suite = unittest.defaultTestLoader.loadTestsFromName("slice_alloc.tests.test_design_contracts")
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    if not result.wasSuccessful():
        print(stream.getvalue())
        raise SystemExit(1)
    files = ["slice_alloc/protocol_contracts.py", "slice_alloc/tools/design_checks.py",
             "slice_alloc/tests/test_design_contracts.py",
             "slice_alloc/tools/platform_verify.py"]
    proof = {
        "passed": True, "tests": result.testsRun,
        "scope": "resource contract kernels and mathematical ranking checks only",
        "diagnostics": expression_diagnostics(),
        "training_performed": False, "packet_simulation_performed": False,
        "new_full_pipeline_implemented": False,
        "source_sha256": {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in files}
    }
    destination = ROOT / "slice_alloc/results/design_verification.json"
    destination.write_text(json.dumps(proof, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps(proof, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
