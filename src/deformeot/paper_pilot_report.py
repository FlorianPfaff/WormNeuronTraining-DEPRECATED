"""Conservative, complete-case feasibility reporting for the bounded pilot."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


def _method_summary(selected, name, stride):
    return {'candidate': name, 'stride': stride, 'recordings': len(selected),
            'mean_accuracy': float(np.mean([row['identity_accuracy'] for row in selected])),
            'mean_switch_rate': float(np.mean([row['identity_switch_rate'] for row in selected])),
            'mean_wrong_duration_scans': float(np.mean([row['mean_wrong_identity_duration_selected_scans'] for row in selected])),
            'sum_runtime_seconds': float(sum(row['runtime_seconds'] for row in selected)),
            'runtime_source': 'concurrent_benchmark_algorithm_reported_not_isolated',
            'future_selected_scans': max(row['future_selected_scans'] for row in selected)}


def summarize(rows, recordings, config):
    benchmark = [row['payload'] for row in rows if row.get('phase') == 'benchmark'
                 and row.get('status') == 'complete']
    names = sorted({row['candidate'] for row in benchmark})
    per_stride = []
    for name in names:
        for stride in config.get('strides', [1]):
            selected = [row for row in benchmark if row['candidate'] == name and row['stride'] == stride]
            complete = len(selected) == len(recordings) and {row['recording_id'] for row in selected} == set(recordings)
            if recordings and complete:
                per_stride.append(_method_summary(selected, name, stride))
    native = [row for row in per_stride if row['stride'] == 1]
    strong = [row for row in native if row['candidate'] in ('cpd_reference', 'cpd_temporal', 'fdnc_reference')]
    comparator = min(strong, key=lambda row: (-row['mean_accuracy'], row['mean_switch_rate'], row['candidate'])) if strong else None
    dual = next((row for row in native if row['candidate'] == 'dual_anchor'), None)
    screen = {'available': bool(dual and comparator), 'primary_stride': 1,
              'interpretation': 'descriptive screen on three purposively selected previously examined animals; no significance claim'}
    if dual and comparator:
        rate = comparator['mean_switch_rate']
        reduction = 1 - dual['mean_switch_rate'] / rate if rate > 0 else None
        accuracy_delta = dual['mean_accuracy'] - comparator['mean_accuracy']
        screen.update(comparator=comparator['candidate'], accuracy_difference=accuracy_delta,
                      relative_switch_rate_reduction=reduction,
                      target_met=reduction is not None and reduction >= config['screen']['minimum_switch_reduction_fraction']
                      and accuracy_delta >= -config['screen']['maximum_accuracy_loss'])
    return {'native_rate_methods': native, 'per_stride_methods': per_stride,
            'screen': screen, 'complete_benchmark_tasks': len(benchmark)}


def summarize_timing(rows, candidates, recordings, budget=None):
    """Use exactly two isolated native-rate repeats per available candidate.

No partial-pair averages are reported. Failed, missing, duplicated, malformed,
or off-recording rows remain explicit rather than being silently discarded.
"""
    target = recordings[1] if len(recordings) == 3 else None
    timing_rows = [row for row in rows if row.get('phase') == 'timing']
    ledger_jobs = (budget or {}).get('jobs', {})
    methods, issues, details = [], [], []
    complete_repeats = 0
    for name in sorted(candidate['name'] for candidate in candidates):
        valid = []
        for repeat in (1, 2):
            identifier = f'timing__{target}__s1__{name}__repeat{repeat}'
            matches = [row for row in timing_rows if row.get('task_id') == identifier]
            if len(matches) != 1:
                issues.append({'candidate': name, 'repeat': repeat,
                               'status': 'missing' if not matches else 'duplicate', 'task_id': identifier})
                continue
            row = matches[0]
            if row.get('status') != 'complete':
                issues.append({'candidate': name, 'repeat': repeat, 'status': row.get('status', 'unknown'),
                               'task_id': identifier, 'error': row.get('error')})
                continue
            payload = row.get('payload', {})
            ledger = ledger_jobs.get(identifier, {})
            runtime, wall = payload.get('runtime_seconds'), ledger.get('wall_seconds')
            coordinates_match = (payload.get('recording_id') == target and payload.get('stride') == 1
                                 and payload.get('candidate') == f'{name}__repeat{repeat}')
            timing_valid = all(isinstance(value, (int, float)) and not isinstance(value, bool)
                               and np.isfinite(value) and value >= 0 for value in (runtime, wall))
            if ledger.get('status') != 'complete':
                issues.append({'candidate': name, 'repeat': repeat,
                               'status': 'missing_or_incomplete_budget_wall_time', 'task_id': identifier})
                continue
            if target is None or not coordinates_match or not timing_valid:
                issues.append({'candidate': name, 'repeat': repeat, 'status': 'invalid', 'task_id': identifier})
                continue
            entry = {'candidate': name, 'recording_id': target, 'stride': 1, 'repeat': repeat,
                     'algorithm_reported_runtime_seconds': float(runtime), 'worker_wall_seconds': float(wall),
                     'worker_reported_interval_seconds': row.get('wall_seconds'),
                     'host_load_start': row.get('host_load_start'), 'host_load_end': row.get('host_load_end')}
            details.append(entry)
            valid.append(entry)
            complete_repeats += 1
        if len(valid) == 2:
            methods.append({'candidate': name, 'recording_id': target, 'stride': 1, 'repeats': 2,
                            'mean_algorithm_reported_runtime_seconds': float(np.mean([row['algorithm_reported_runtime_seconds'] for row in valid])),
                            'mean_worker_wall_seconds': float(np.mean([row['worker_wall_seconds'] for row in valid]))})
    dual = next((row for row in methods if row['candidate'] == 'dual_anchor'), None)
    for row in methods:
        for metric, ratio in (('mean_algorithm_reported_runtime_seconds', 'algorithm_reported_ratio_to_dual_anchor'),
                              ('mean_worker_wall_seconds', 'worker_wall_ratio_to_dual_anchor')):
            row[ratio] = row[metric] / dual[metric] if dual and dual[metric] > 0 else None
    expected = len(candidates) * 2
    return {'recording_id': target, 'stride': 1, 'repeats_per_method': 2,
            'expected_repeats_for_available_candidates': expected,
            'expected_repeats_all_prespecified_methods': 20,
            'complete_repeats': complete_repeats,
            'complete': bool(expected and target is not None and complete_repeats == expected and not issues),
            'methods': methods, 'repeat_details': details, 'issues': issues,
            'interpretation': 'Serial repeat observations on one median-motion animal; not uniformly instrumented steady-state or end-to-end tracker latency. Ratios are descriptive, not definitive speedups.'}


def feasibility_decision(screen, blockers):
    """Keep a negative practical screen distinct from unresolved readiness."""
    if screen.get('available') and not screen.get('target_met'):
        return 'stop_superiority_route_under_current_design', 'failed'
    if blockers or not screen.get('available'):
        return 'conditional_go', 'passed' if screen.get('available') else 'not_evaluable'
    return 'go_to_independent_validation', 'passed'


def summarize_historical_audit(audit):
    comparisons = [case.get('comparisons', {}).get('current_vs_historical_reconstruction', {})
                   for case in audit.get('cases', [])]
    available = [row for row in comparisons if row.get('available')]
    identical = sum(row.get('assignments_identical') is True for row in available)
    captures = sum(bool(row.get('diagnostic_arrays')) for row in available)
    if len(available) == 6 and identical == 6:
        conclusion = ('The current and reconstructed historical environments produced identical assignments '
                      'in all six diagnostic cases. There were no assignment divergences and therefore '
                      'no first-divergence captures for that environment contrast.')
    else:
        conclusion = (f'The current-versus-reconstructed environment contrast is available for {len(available)} '
                      f'of six cases, with identical assignments in {identical}; {captures} contrasts have '
                      'first-divergence captures. These concern present-day reruns, not the original July execution.')
    if not audit.get('historical_assignment_arrays_available', False):
        conclusion += (' Original July assignment arrays are unavailable, so the original divergence '
                       'cannot be located and the historical discrepancy remains unexplained.')
    return {'available_environment_contrasts': len(available), 'identical_assignment_cases': identical,
            'contrasts_with_first_divergence_captures': captures,
            'historical_assignment_arrays_available': audit.get('historical_assignment_arrays_available', False),
            'conclusion': conclusion}


def _write_csv(path, rows):
    if not rows:
        path.write_text('')
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
                         for key, value in row.items()} for row in rows)


def generate_report(root, output, config):
    output, root = Path(output), Path(root)
    def load(path, default=None):
        return json.loads(path.read_text()) if path.exists() else default
    rows = [json.loads(path.read_text()) for path in sorted((output / 'checkpoints').glob('*.json'))]
    selection = load(output / 'pilot_selection.json', {})
    recordings = selection.get('selected_recordings', [])
    analysis = summarize(rows, recordings, config)
    audit = load(output / 'audit/numerical_audit.json', {})
    inventory = load(output / 'inventory/inventory.json', {})
    validation = load(output / 'checkpoints/validate.json', {})
    tests = load(output / 'checkpoints/tests.json', {})
    final_verification = load(output / 'final_verification.json', {})
    budget = load(output / 'budget.json', {})
    candidates = load(output / 'candidates.json', {}).get('candidates', [])
    timing = summarize_timing(rows, candidates, recordings, budget)
    historical_audit = summarize_historical_audit(audit)
    names = {row['name'] for row in candidates}
    blockers = []
    if tests.get('status') != 'complete':
        blockers.append('The full verification suite has not passed.')
    if final_verification and final_verification.get('status') != 'passed':
        blockers.append('The reporting revision has not passed final verification.')
    if validation.get('payload', {}).get('status') != 'passed':
        blockers.append('Learned fDNC adapter fidelity has not passed validation.')
    if not {'cpd_reference', 'cpd_temporal', 'fdnc_reference'} <= names:
        blockers.append('At least one prespecified strong baseline is missing.')
    if not audit.get('all_available_environments_deterministic') or not audit.get('all_requested_environments_available'):
        blockers.append('Numerical reproducibility auditing is incomplete or failed.')
    if not audit.get('all_input_provenance_checks_pass') or not audit.get('all_recorded_source_hashes_match'):
        blockers.append('Audit input/source provenance has not fully passed.')
    if not audit.get('scoring_fixtures', {}).get('all_passed'):
        blockers.append('Identity-scoring fixtures have not fully passed.')
    if inventory.get('external_cohort_status') != 'eligible':
        blockers.append('No independently annotated external multi-animal cohort has yet been fully verified.')
    # A reconstructed environment is not equivalent to the original host. Do not
    # turn deterministic reruns into a claim that historical discrepancies were solved.
    historical_matches = [all(case.get('runs', {}).get('historical_reconstruction', {}).get('historical_metrics_match', []))
                          and len(case.get('runs', {}).get('historical_reconstruction', {}).get('historical_metrics_match', [])) == 3
                          for case in audit.get('cases', [])]
    if len(historical_matches) != 6 or not all(historical_matches):
        blockers.append('Not every selected original historical aggregate result was reproduced. '
                        'Present-day environment agreement does not resolve the original discrepancy; '
                        'the original July assignment arrays are unavailable.')
    expected = len(recordings) * len(config['strides']) * len(candidates)
    if not expected or analysis['complete_benchmark_tasks'] != expected:
        blockers.append('The planned pilot comparison is incomplete; missing cases were not treated as successful.')
    if not timing['complete']:
        blockers.append(f"Isolated timing repeats are incomplete: {timing['complete_repeats']} / "
                        f"{timing['expected_repeats_for_available_candidates']} for available methods; "
                        'missing, failed, or invalid repeats are retained in the timing issues.')
    if budget.get('terminal_reason'):
        blockers.append(f"Execution stopped at the resource limit: {budget['terminal_reason']}.")
    screen = analysis['screen']
    decision, practical_status = feasibility_decision(screen, blockers)
    benchmark_complete = bool(expected and analysis['complete_benchmark_tasks'] == expected)
    required_phases_complete = all(any(row.get('phase') == phase and row.get('status') == 'complete'
                                       for row in rows) for phase in ('audit', 'inventory', 'tests', 'validate'))
    result = {'status': 'complete' if benchmark_complete and timing['complete'] and required_phases_complete else 'partial',
              'decision': decision, 'blockers': blockers, 'analysis': analysis,
              'practical_screen_outcome': practical_status,
              'publication_readiness': 'blocked' if blockers else 'ready_for_independent_validation' if practical_status == 'passed' else 'screen_failed',
              'benchmark_status': 'complete' if benchmark_complete else 'partial',
              'isolated_timing': timing, 'historical_environment_comparison': historical_audit,
              'pilot_selection': selection, 'budget': budget,
              'reporting_revision_verification': final_verification,
              'expected_benchmark_tasks_for_available_candidates': expected,
              'expected_benchmark_tasks_all_prespecified_methods': 90,
              'historical_reconstruction_matching_cases': sum(historical_matches),
              'reference_identity_classification': inventory.get('reference_identity_classification'),
              'external_cohort_status': inventory.get('external_cohort_status', 'unverified'),
              'no_new_untouched_outcomes_evaluated': True,
              'failed_tasks': [{'task_id': row['task_id'], 'status': row['status'], 'error': row.get('error')}
                               for row in rows if row.get('status') != 'complete']}
    destination = root / 'outputs/paper_pilot_20261003'
    destination.mkdir(parents=True, exist_ok=True)
    (destination / 'summary.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    _write_csv(destination / 'native_rate_comparison.csv', analysis['native_rate_methods'])
    _write_csv(destination / 'per_stride_comparison.csv', analysis['per_stride_methods'])
    _write_csv(destination / 'benchmark_details.csv', [
        {'task_id': row.get('task_id'), 'status': row.get('status'), 'error': row.get('error'),
         'worker_reported_interval_seconds': row.get('wall_seconds'), **row.get('payload', {})}
        for row in rows if row.get('phase') == 'benchmark'])
    _write_csv(destination / 'isolated_timing.csv', timing['methods'])
    _write_csv(destination / 'isolated_timing_repeats.csv', timing['repeat_details'])
    _write_csv(destination / 'isolated_timing_issues.csv', timing['issues'])
    lines = ['# Neuron tracking paper feasibility pilot', '',
             f"Decision: **{decision.replace('_', ' ')}**.", '',
             f"Practical screen: **{practical_status.replace('_', ' ')}**. "
             f"Publication readiness: **{result['publication_readiness'].replace('_', ' ')}**. "
             'A failed development screen is not converted into a positive go because other evidence remains unresolved.', '',
             'This is a bounded exploratory development pilot, not independent confirmation. '
             'Three recordings were selected by low, median, and high identity-blind motion; '
             'the original tuning recording was excluded. All recordings had been examined in earlier work.', '',
             '## Native rate comparison', '',
             '| Method | Accuracy | Switches per 1000 neuron-transitions | Future scans |',
             '| --- | ---: | ---: | ---: |']
    for row in analysis['native_rate_methods']:
        lines.append(f"| {row['candidate']} | {100*row['mean_accuracy']:.3f}% | "
                     f"{1000*row['mean_switch_rate']:.3f} | {row['future_selected_scans']} |")
    if screen.get('available'):
        reduction = screen['relative_switch_rate_reduction']
        lines += ['', f"Development-selected strongest new comparator by native-rate accuracy: `{screen['comparator']}`. "
                  f"Dual-anchor accuracy difference: {100*screen['accuracy_difference']:+.3f} percentage points.",
                  f"Relative switch-rate reduction: {100*reduction:+.1f}%." if reduction is not None else
                  'The selected comparator has zero switches, so relative switch reduction is undefined.',
                  f"The provisional 25% switch-reduction / at-most-0.25-point accuracy-loss screen passed: `{screen['target_met']}`."]
    lines += ['', 'Method means weight the three recordings equally. Native rate is the primary screen; '
              'stride-10 and stride-40 means are stress-test summaries in `per_stride_comparison.csv`. '
              'Per-case metrics and failures are retained in `benchmark_details.csv`.', '',
              '`mht_geometry_online` has lag zero: its result reflects Gaussian filtering/scoring, '
              'not persistent competing association histories. `mht_geometry_lag2` uses two future '
              'selected observations; it is not a matched-information comparison with an online method.', '',
              '## Isolated repeated timing', '',
              f"Median-motion recording: `{timing['recording_id']}` at stride 1. "
              f"Complete repeats: {timing['complete_repeats']} / {timing['expected_repeats_for_available_candidates']} "
              'for available methods (20 planned across all ten methods). Each reported method has both repeats.', '',
              '| Method | Algorithm-reported mean (s) | Ratio to dual | Whole-worker mean (s) | Ratio to dual |',
              '| --- | ---: | ---: | ---: | ---: |']
    for row in timing['methods']:
        algorithm_ratio = row['algorithm_reported_ratio_to_dual_anchor']
        worker_ratio = row['worker_wall_ratio_to_dual_anchor']
        algorithm_label = f'{algorithm_ratio:.2f}' if algorithm_ratio is not None else 'unavailable'
        worker_label = f'{worker_ratio:.2f}' if worker_ratio is not None else 'unavailable'
        lines.append(f"| {row['candidate']} | {row['mean_algorithm_reported_runtime_seconds']:.3f} | "
                     f"{algorithm_label} | {row['mean_worker_wall_seconds']:.3f} | {worker_label} |")
    lines += ['', 'These are serial repeated observations with no other pilot jobs running, but unrelated '
              'server load remained uncontrolled and is recorded per repeat. Algorithm-reported timer '
              'boundaries differ: the fDNC timer starts after model loading and preprocessing and includes '
              'inference plus the motion metric; other adapters have different boundaries. Whole-worker '
              'wall time comes from the supervisor budget ledger and includes process startup, imports/setup, '
              'loading, tracking and scoring as a separate process-level proxy; checkpoint-reported worker '
              'intervals, which exclude initial startup/imports, are also retained in the repeat CSV. '
              'Neither column is uniformly instrumented steady-state or end-to-end tracker latency. '
              'Ratios are descriptive observations, not definitive speedups or the best attainable fDNC GPU performance. '
              'No model-load/steady-state decomposition was measured.', '',
              '`isolated_timing.csv` contains complete-pair means and ratios; `isolated_timing_repeats.csv` '
              'retains individual observations and load. Concurrent benchmark runtimes are retained separately '
              'and are not used for this timing table.']
    lines += ['', '## Audit and limitations', '']
    lines += [f'- {item}' for item in blockers] or ['- No automatic feasibility blockers were detected; this is not an acceptance guarantee.']
    lines += ['', f"Reference identity classification: {inventory.get('reference_identity_classification', 'pending')}.",
              'Accuracy means agreement with supplied ROI trajectories. The reviewed acquisition workflow '
              'used algorithm-assisted tracking followed by expert verification; independent manual identity '
              'annotation at every frame was not established. Stable ROI rows and semantic NeuroPAL names '
              'are distinct from independently verified biological identity ground truth.',
              '', f"Complete benchmark tasks: {analysis['complete_benchmark_tasks']} / {expected} for available candidates "
              '(90 planned with all ten methods).',
              f"Conservatively accounted cumulative job time: {budget.get('completed_job_seconds', 0)/3600:.3f} / 24 hours. "
              f"Accounted additional storage: {budget.get('storage_bytes', 0)/1024**3:.3f} / 10 GiB.", '',
              'Historical environment reconstruction matches package pins where possible, not the original '
              'Python patch version, WSL host, or BLAS dispatch. All recorded mismatches are retained. '
              + historical_audit['conclusion'] + ' Do not attribute the original discrepancies to ties or '
              'dependencies without further evidence.', '',
              'Identity accuracy includes the initialized frame for compatibility with historical scores; '
              'the machine-readable rows also give accuracy excluding it. All methods have complete '
              'one-to-one coverage. Wrong-identity episode lengths are measured in selected scans, not seconds. '
              'No detector-gap, extent, calibrated-uncertainty, or downstream calcium claim is evaluated.', '',
              'This implements only the bounded first work package. Matched-lookahead mechanism '
              'ablations, new-source confirmation, biological utility, and a comprehensive published-method '
              'ranking remain outside this run. The five-frame fDNC inference check validates adapter '
              'fidelity, not training-animal independence. Peak memory, warmup-adjusted timing and '
              'annotation effort were not measured.', '',
              'The public-source metadata review was not individually timed; its five short server jobs '
              'are conservatively charged at 1,200 seconds. Other numerical worker jobs are wall-clock timed.', '',
              '## Reproduction', '',
              'Run `python -m deformeot.paper_pilot --help` for phase controls. The experiment uses '
              '`configs/paper_pilot.yaml`, separate per-task checkpoints, a locked resource ledger, '
              'and exact data/code/model/environment signatures. Historical frozen files are not edited.', '',
              'The detailed numerical audit, baseline validation, inventory evidence, task logs, '
              'and budget are retained under `work/results/paper_pilot_20261003`. Failed or missing '
              'baselines are not silently replaced. No external test identities were scored.', '']
    if final_verification:
        lines += ['The exact execution source is retained in `execution_source.tar.gz` beside `context.json`. '
                  'A reporting-only revision adds timing summaries without changing the numerical checkpoints; '
                  '`final_verification.json` records both renderer hashes, the source archive hash, and final '
                  'verification. Its enforced two-minute reservation is charged conservatively to the budget. '
                  'Supervisor wall time is additionally charged as a conservative overhead allowance; '
                  'the total is not a CPU-time measurement.', '']
    (destination / 'report.md').write_text('\n'.join(lines))
    return {'report': str(destination / 'report.md'), 'summary': str(destination / 'summary.json'),
            'decision': decision, 'complete_benchmark_tasks': analysis['complete_benchmark_tasks'], 'blockers': blockers}
