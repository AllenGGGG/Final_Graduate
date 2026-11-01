"""Read-only check: does sampled rollout agree with DDQN's active-rejection timing?

Never changes frozen training, calibration, policies or packet outcomes. Replays
each frozen DDQN decision; at every policy-rejection point, additionally asks the
sampled-rollout comparator (joint_rollout.sampled_action) what it would have done
in that exact replayed state. This is a comparator agreement check, not a causal
attribution of the stability gap, and not a claim that rollout is a better teacher.
"""
import json
import numpy as np
import torch
from ..joint import JointEnv, request_hash
from ..src.qlearning import DoubleDQN
from .joint_run import OUT, profile, source_hashes, write
from .joint_rollout import sampled_action
from .platform_verify import sha


def summarize_agreement(rows, success_rates):
    """Pool decisions with their actual denominators; correlate only by model."""
    counts = {method: dict(policy_rejections=0, rollout_also_rejects=0,
                          rollout_accepts=0) for method in success_rates}
    seen = set()
    for row in rows:
        key = (row['case'], row['id'])
        if key in seen:
            raise ValueError('duplicate rejection decision')
        seen.add(key)
        current = counts[row['method']]
        agrees = row['rollout_agrees_with_rejection']
        if not isinstance(agrees, bool):
            raise ValueError('agreement must be boolean')
        current['policy_rejections'] += 1
        current['rollout_also_rejects'] += int(agrees)
        current['rollout_accepts'] += int(not agrees)
    summary = {}
    for method, current in sorted(counts.items()):
        n = current['policy_rejections']
        summary[method] = dict(**current,
            rollout_agreement_rate=current['rollout_also_rejects'] / n if n else None,
            packet_success_rate=success_rates[method])
    total = sum(x['policy_rejections'] for x in summary.values())
    agreed = sum(x['rollout_also_rejects'] for x in summary.values())
    paired = [x for x in summary.values() if x['policy_rejections']]
    x = [r['rollout_agreement_rate'] for r in paired]
    y = [r['packet_success_rate'] for r in paired]
    correlation = (float(np.corrcoef(x, y)[0, 1])
                   if len(x) > 1 and np.std(x) > 0 and np.std(y) > 0 else None)
    return summary, dict(policy_rejections=total, rollout_also_rejects=agreed,
        rollout_accepts=total-agreed,
        rollout_agreement_rate=agreed / total if total else None,
        rollout_disagreement_rate=(total-agreed) / total if total else None,
        pearson_r_across_initializations=correlation,
        correlation_initializations=len(paired))


def main():
    torch.set_num_threads(1)
    prof = profile()
    stability_path = OUT / 'stability_comparison.json'
    data = json.loads(stability_path.read_text())
    training = json.loads((OUT / 'stability.json').read_text())
    if data['comparison_sha256'] != sha(OUT / 'comparison.json') or data['training_sha256'] != sha(OUT / 'stability.json'):
        raise ValueError('stale stability input')
    models = {}
    for record in training['models']:
        path = OUT / record['file']
        if sha(path) != record['sha256']:
            raise ValueError('model mismatch')
        if record['algorithm'] == 'ddqn':
            models[f"ddqn_{record['initialization']}"] = DoubleDQN.load(path)
    rollout_config = json.loads((OUT / 'rollout_comparison.json').read_text())['selected_config']

    agreement_rows = []
    for row in data['episodes']:
        method = row['method']
        model = models.get(method)
        if model is None:
            continue
        env = JointEnv(prof, count=row['requests'], load=row['load'])
        obs, _ = env.reset(seed=row['seed'])
        if request_hash(env.requests) != row['input_sha256']:
            raise ValueError('replay request mismatch')
        for original in row['decisions']:
            action = int(model.predict(obs, deterministic=True)[0])
            selected = env.plans[action - 1].kind if action else None
            if selected != original['selected']:
                raise ValueError('frozen decision not reproduced')
            if env.plans and not action:
                rollout_choice = sampled_action(env, **rollout_config)
                agrees = rollout_choice == 0
                agreement_rows.append(dict(case=row['case'], method=method, id=original['id'],
                    arrival=original['arrival'], rollout_agrees_with_rejection=agrees))
            obs, _, _, _, _ = env.step(action)

    success_rates = {f"ddqn_{r['initialization']}": r['success_rate']
                     for r in data['summary']['ddqn']['per_initialization']}
    summary, aggregate = summarize_agreement(agreement_rows, success_rates)
    proof = dict(passed=True, source_sha256=sha(__file__), frozen_source_sha256=source_hashes(),
        stability_comparison_sha256=sha(stability_path),
        rollout_comparison_sha256=sha(OUT / 'rollout_comparison.json'),
        rollout_config=rollout_config,
        per_initialization_agreement=summary,
        aggregate=aggregate,
        agreement_rows=agreement_rows,
        scope=('agreement between DDQN active-rejection timing and the sampled-rollout comparator '
               'at the exact same replayed state; rollout uses training-law synthetic futures, not '
               'the realized held-out future; disagreement does not prove either side is correct, '
               'and this does not decompose the 8-seed stability gap into causal shares; '
               'only DDQN rejection states are sampled, not both policies full decision distributions; '
               f'the {len(agreement_rows)} active-rejection samples share trajectories and '
               f'the correlation uses {aggregate["correlation_initializations"]} initializations with rejection samples; '
               'Q estimation and modality preferences are not excluded as explanations'))
    write(OUT / 'reject_timing_agreement.json', proof)
    print(json.dumps(dict(summary=summary, aggregate=aggregate), ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
