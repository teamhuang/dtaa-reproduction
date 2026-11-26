"""DTAA模拟基准: S1-S4已知真值情景, 验证审计程序按预设规则正确识别已知问题.
运行: python -m pytest test_benchmark.py -v  或直接 python test_benchmark.py
"""
import pandas as pd

from dtaa_audit.core import load_spec, build_registry, compute_indices, detect_issues


def make_sim_spec(scenario: str) -> pd.DataFrame:
    """构造已知真值的规范. 真值在注释中, 断言在测试中."""
    rows = []
    def add(vid, name, tc, shared, c, role, in_model=0, patient_key=1):
        rows.append({'variable_id': vid, 'dataset': 'SIM', 'raw_column': name, 'source_table': 'main',
                     'timing_class': tc, 'shared': shared, 'clinically_accessible': c,
                     'causal_role': role, 'missing_pct': 0.0, 'in_model': in_model, 'patient_key': patient_key})
    add('X001', 'age', 'T0_available', 1, 1, 'candidate_predictor', 1)
    add('X002', 'severity', 'T0_available', 1, 1, 'candidate_predictor', 1)
    if scenario == 'S1':      # 治疗后变量被纳入T0模型
        add('X003', 'post_vital', 'post_decision', 1, 0, 'post_decision_predictor', 1)
        add('X004', 'biomarker', 'T0_available', 1, 1, 'candidate_predictor', 1)
    elif scenario == 'S2':    # 中介变量用于总效应估计
        add('X003', 'mediator_v', 'post_decision', 1, 0, 'mediator', 1)
        add('X004', 'biomarker', 'T0_available', 1, 1, 'candidate_predictor', 1)
    elif scenario == 'S3':    # 决策前变量已测量但未共享
        add('X003', 'prior_score', 'unobserved', 0, 1, 'confounder_unshared', 0)
        add('X004', 'biomarker', 'T0_available', 1, 1, 'candidate_predictor', 1)
    elif scenario == 'S4':    # 患者内聚类, 患者键缺失
        add('X003', 'lesion_2', 'T0_available', 1, 1, 'candidate_predictor', 1, patient_key=0)
        add('X004', 'biomarker', 'T0_available', 1, 1, 'candidate_predictor', 1)
    add('Y001', 'outcome_y', 'outcome', 1, 0, 'outcome')
    return pd.DataFrame(rows)


def run(scenario):
    reg = build_registry(load_spec(make_sim_spec(scenario)))
    idx = compute_indices(reg, 'SIM', 'T0')
    issues = detect_issues(reg)
    return reg, idx, issues


def test_S1_post_decision_detection():
    reg, idx, issues = run('S1')
    types = [i['type'] for i in issues]
    assert 'S1_post_decision_in_model' in types, issues
    assert idx['PAVR'] == 0.25                      # 模型4协变量中1个决策后
    assert idx['DAI_C'] == 0.75                     # 合格4个, 3个T0可得
    print('S1 PASS: 决策后变量入模型被检出, PAVR=1/4, DAI_C=3/4')


def test_S2_mediator_detection():
    reg, idx, issues = run('S2')
    types = [i['type'] for i in issues]
    assert 'S2_mediator_for_total_effect' in types, issues
    print('S2 PASS: 中介变量用于总效应估计被检出')


def test_S3_unshared_detection():
    reg, idx, issues = run('S3')
    types = [i['type'] for i in issues]
    assert 'S3_measured_not_shared' in types, issues
    assert idx['DAI_C'] > idx['DAI_D'], (idx['DAI_C'], idx['DAI_D'])  # 临床可得>共享可得
    assert idx['DAI_C'] == 1.0 and idx['DAI_D'] == 0.75
    print(f'S3 PASS: missing column detected, DAI_C={idx["DAI_C"]}, DAI_D={idx["DAI_D"]}')


def test_S4_patient_key():
    reg, idx, issues = run('S4')
    types = [i['type'] for i in issues]
    assert 'S4_patient_key_missing' in types, issues
    print('S4 PASS: 患者键缺失被检出')


def test_decision_point_switch():
    """决策点切换: T1时T1_available变量变为可得, DAI更新."""
    spec = make_sim_spec('S1')
    extra = pd.DataFrame([{'variable_id': 'X005', 'dataset': 'SIM', 'raw_column': 'followup_lab',
                           'source_table': 'main', 'timing_class': 'T1_available', 'shared': 1,
                           'clinically_accessible': 1, 'causal_role': 'post_decision_predictor',
                           'missing_pct': 0.0, 'in_model': 0, 'patient_key': 1}])
    reg = build_registry(load_spec(pd.concat([spec, extra])))
    i0 = compute_indices(reg, 'SIM', 'T0')
    i1 = compute_indices(reg, 'SIM', 'T1')
    assert i0['DAI_C'] < i1['DAI_C'], (i0['DAI_C'], i1['DAI_C'])
    assert i1['DAI_C'] == 1.0   # T1时全部5个合格变量可得
    print(f'决策点切换 PASS: DAI_C {i0["DAI_C"]}(T0) -> {i1["DAI_C"]}(T1)')


def test_index_correctness():
    """指标计算正确性: 与手工真值对照."""
    reg, idx, _ = run('S1')
    # 合格=4 (age, severity, post_vital, biomarker), T0可得=3
    assert idx['n_eligible'] == 4
    assert idx['DAI_D'] == 0.75
    assert idx['PUR'] == 0.0
    print('指标正确性 PASS')


def test_reproducibility():
    """审计可重复性: 同输入两次运行结果一致."""
    r1 = build_registry(load_spec(make_sim_spec('S3')))
    r2 = build_registry(load_spec(make_sim_spec('S3')))
    pd.testing.assert_frame_equal(r1, r2)
    print('可重复性 PASS')


if __name__ == '__main__':
    test_S1_post_decision_detection()
    test_S2_mediator_detection()
    test_S3_unshared_detection()
    test_S4_patient_key()
    test_decision_point_switch()
    test_index_correctness()
    test_reproducibility()
    print('\n全部7项基准测试通过')
