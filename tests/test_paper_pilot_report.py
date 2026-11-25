import json

import pytest

from deformeot.paper_pilot_report import (
    feasibility_decision,
    generate_report,
    summarize,
    summarize_historical_audit,
    summarize_timing,
)


RECORDINGS = ['low', 'median', 'high']
METHODS = ['previous_hungarian', 'pairwise_affine', 'first_affine', 'dual_anchor',
           'cpd_reference', 'cpd_temporal', 'fdnc_reference', 'gnn_kalman',
           'mht_geometry_online', 'mht_geometry_lag2']


def timing_fixture(methods=METHODS):
    rows, jobs = [], {}
    for index, name in enumerate(methods):
        for repeat in (1, 2):
            identifier = f'timing__median__s1__{name}__repeat{repeat}'
            runtime = float((index + 1) * repeat)
            rows.append({'phase': 'timing', 'task_id': identifier, 'status': 'complete',
                         'wall_seconds': runtime + 4,
                         'host_load_start': [20., 25., 30.], 'host_load_end': [21., 25., 30.],
                         'payload': {'candidate': f'{name}__repeat{repeat}', 'recording_id': 'median',
                                     'stride': 1, 'runtime_seconds': runtime}})
            jobs[identifier] = {'status': 'complete', 'wall_seconds': runtime + 5}
    return rows, [{'name': name} for name in methods], {'jobs': jobs}


def benchmark_row(name, recording, stride=1, accuracy=.95, rate=.01):
    return {'phase': 'benchmark', 'status': 'complete',
            'task_id': f'benchmark__{recording}__s{stride}__{name}',
            'payload': {'candidate': name, 'recording_id': recording, 'stride': stride,
                        'identity_accuracy': accuracy, 'identity_switch_rate': rate,
                        'mean_wrong_identity_duration_selected_scans': 2.,
                        'runtime_seconds': 1., 'future_selected_scans': 0}}


def test_isolated_timing_requires_twenty_complete_repeats_and_uses_supervisor_wall():
    rows, candidates, budget = timing_fixture()
    result = summarize_timing(rows, candidates, RECORDINGS, budget)
    assert result['complete']
    assert result['complete_repeats'] == result['expected_repeats_all_prespecified_methods'] == 20
    assert len(result['methods']) == 10
    assert not result['issues']
    dual = next(row for row in result['methods'] if row['candidate'] == 'dual_anchor')
    assert dual['mean_algorithm_reported_runtime_seconds'] == 6.
    assert dual['mean_worker_wall_seconds'] == 11.
    assert dual['algorithm_reported_ratio_to_dual_anchor'] == 1.
    previous = next(row for row in result['methods'] if row['candidate'] == 'previous_hungarian')
    assert previous['algorithm_reported_ratio_to_dual_anchor'] == .25
    assert previous['worker_wall_ratio_to_dual_anchor'] == pytest.approx(6.5 / 11.)
    detail = result['repeat_details'][0]
    assert detail['worker_wall_seconds'] == detail['worker_reported_interval_seconds'] + 1


def test_missing_and_failed_timing_repeats_exclude_partial_method_means():
    rows, candidates, budget = timing_fixture(['dual_anchor', 'cpd_reference', 'fdnc_reference'])
    rows = [row for row in rows if row['task_id'] != 'timing__median__s1__cpd_reference__repeat2']
    failed = next(row for row in rows if row['task_id'] == 'timing__median__s1__fdnc_reference__repeat1')
    failed.update(status='failed', error='fixture failure')
    result = summarize_timing(rows, candidates, RECORDINGS, budget)
    assert not result['complete']
    assert result['complete_repeats'] == 4
    assert [row['candidate'] for row in result['methods']] == ['dual_anchor']
    assert {row['status'] for row in result['issues']} == {'missing', 'failed'}


def test_timing_does_not_substitute_concurrent_benchmarks_or_worker_intervals():
    rows, candidates, budget = timing_fixture(['dual_anchor'])
    rows.append(benchmark_row('dual_anchor', 'median'))
    budget['jobs'].pop('timing__median__s1__dual_anchor__repeat2')
    result = summarize_timing(rows, candidates, RECORDINGS, budget)
    assert not result['complete']
    assert result['methods'] == []
    assert result['issues'][0]['status'] == 'missing_or_incomplete_budget_wall_time'


def test_duplicate_and_wrong_recording_timing_rows_are_invalid():
    rows, candidates, budget = timing_fixture(['dual_anchor'])
    rows.append(dict(rows[0]))
    rows[1]['payload']['recording_id'] = 'low'
    result = summarize_timing(rows, candidates, RECORDINGS, budget)
    assert {row['status'] for row in result['issues']} == {'duplicate', 'invalid'}
    assert not result['methods']


def test_failed_practical_screen_is_not_promoted_by_readiness_blockers():
    decision, screen = feasibility_decision({'available': True, 'target_met': False},
                                            ['External data unresolved.', 'Historical drift unresolved.'])
    assert decision == 'stop_superiority_route_under_current_design'
    assert screen == 'failed'


def test_passed_screen_with_blockers_is_only_conditional_and_missing_screen_is_not_evaluable():
    assert feasibility_decision({'available': True, 'target_met': True}, ['Labels unresolved.']) == ('conditional_go', 'passed')
    assert feasibility_decision({'available': False}, []) == ('conditional_go', 'not_evaluable')
    assert feasibility_decision({'available': True, 'target_met': True}, []) == ('go_to_independent_validation', 'passed')


def test_identical_reconstruction_audit_does_not_claim_unavailable_divergence_captures():
    comparison = {'available': True, 'assignments_identical': True,
                  'first_assignment_divergence_selected_frame': None}
    audit = {'historical_assignment_arrays_available': False,
             'cases': [{'comparisons': {'current_vs_historical_reconstruction': comparison}} for _ in range(6)]}
    result = summarize_historical_audit(audit)
    assert result['identical_assignment_cases'] == 6
    assert result['contrasts_with_first_divergence_captures'] == 0
    assert 'no first-divergence captures' in result['conclusion']
    assert 'Original July assignment arrays are unavailable' in result['conclusion']
    assert 'remains unexplained' in result['conclusion']


def test_per_stride_averages_require_all_three_animals():
    rows = [benchmark_row('dual_anchor', recording, stride)
            for recording in RECORDINGS for stride in (1, 10, 40)]
    rows.pop()
    config = {'strides': [1, 10, 40],
              'screen': {'minimum_switch_reduction_fraction': .25, 'maximum_accuracy_loss': .0025}}
    result = summarize(rows, RECORDINGS, config)
    assert [row['stride'] for row in result['per_stride_methods']] == [1, 10]
    assert result['native_rate_methods'][0]['runtime_source'] == 'concurrent_benchmark_algorithm_reported_not_isolated'


def test_report_missing_timings_blocks_readiness_and_keeps_failed_screen(tmp_path):
    output = tmp_path / 'results'
    checkpoints = output / 'checkpoints'
    checkpoints.mkdir(parents=True)
    (output / 'pilot_selection.json').write_text(json.dumps({'selected_recordings': RECORDINGS}))
    (output / 'candidates.json').write_text(json.dumps({'candidates': [{'name': 'dual_anchor'}, {'name': 'cpd_reference'}]}))
    for name, accuracy, rate in [('dual_anchor', .95, .02), ('cpd_reference', .99, .01)]:
        for recording in RECORDINGS:
            row = benchmark_row(name, recording, accuracy=accuracy, rate=rate)
            (checkpoints / f"{row['task_id']}.json").write_text(json.dumps(row))
    config = {'strides': [1], 'screen': {'minimum_switch_reduction_fraction': .25, 'maximum_accuracy_loss': .0025}}
    rendered = generate_report(tmp_path, output, config)
    summary = json.loads((tmp_path / 'outputs/paper_pilot_20261003/summary.json').read_text())
    report = (tmp_path / 'outputs/paper_pilot_20261003/report.md').read_text()
    assert rendered['decision'] == 'stop_superiority_route_under_current_design'
    assert summary['practical_screen_outcome'] == 'failed'
    assert summary['status'] == 'partial'
    assert summary['publication_readiness'] == 'blocked'
    assert any('Isolated timing repeats are incomplete' in blocker for blocker in summary['blockers'])
    assert 'per 1000 neuron-transitions' in report
    assert 'not persistent competing association histories' in report
    assert 'independent manual identity annotation at every frame was not established' in report
    assert (tmp_path / 'outputs/paper_pilot_20261003/benchmark_details.csv').exists()
    assert (tmp_path / 'outputs/paper_pilot_20261003/isolated_timing_issues.csv').exists()
