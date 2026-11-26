"""DTAA审计工具核心: 三层T/R/C判定 + 指标计算 + Markdown审计报告.
输入规范(Spec)列:
  variable_id, dataset, raw_column, source_table,
  timing_class (T0_available/T1_available/T1_intra/post_decision/timing_unclear/outcome/identifier/unobserved),
  shared (1/0), clinically_accessible (1/0/U), causal_role, missing_pct, in_model (0/1)
"""
from __future__ import annotations
import pandas as pd

TIMING_MAP = {
    # timing_class -> (T at primary decision point, D1 label)
    'T0_available': (1, 'T0_available'),
    'T1_available': (0, 'T1_available'),        # 主决策点不可得, 后续决策点可得
    'T1_intra': (0, 'T1_intra'),
    'post_decision': (0, 'T1_post_decision'),
    'timing_unclear': ('U', 'timing_unclear'),
    'outcome': (0, 'outcome'),
    'identifier': ('NA', 'identifier'),
    'unobserved': (1, 'T0_available'),          # 临床已测量但未共享
}
ELIGIBLE_ROLES = {'candidate_predictor', 'post_decision_predictor', 'candidate_timing_unclear',
                  'confounder_unshared', 'intra_procedural'}


def load_spec(path_or_df) -> pd.DataFrame:
    df = path_or_df if isinstance(path_or_df, pd.DataFrame) else pd.read_csv(path_or_df)
    need = {'variable_id', 'dataset', 'raw_column', 'timing_class', 'shared', 'causal_role', 'in_model'}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f'规范缺少必需列: {missing}')
    if 'clinically_accessible' not in df.columns:
        # 默认: T0可得且已共享 -> 可及; 未共享但临床已测量 -> 1; 决策后 -> 0
        def default_c(r):
            t = TIMING_MAP[r['timing_class']][0]
            if t == 1:
                return 1
            if t == 'U':
                return 'U'
            return 0
        df['clinically_accessible'] = df.apply(default_c, axis=1)
    return df


def build_registry(spec: pd.DataFrame) -> pd.DataFrame:
    """展开为审计矩阵: T/R/C三层 + D1-D4."""
    out = spec.copy()
    out[['T_t0', 'D1_temporal']] = out['timing_class'].apply(
        lambda c: pd.Series(TIMING_MAP[c]))
    out['R_shared'] = out['shared'].astype(int)
    out['C_clinical'] = out['clinically_accessible']
    out['D2_accessibility'] = out['C_clinical'].map(
        {1: 'available_at_decision', 0: 'only_after_decision', 'U': 'uncertain'})
    out['D4_reproducibility'] = out.apply(
        lambda r: 'unobserved_in_shared_data' if r['R_shared'] == 0
        else ('documented' if r['missing_pct'] == 0 else 'documented_with_missing'), axis=1)
    return out


def compute_indices(registry: pd.DataFrame, dataset: str | None = None,
                    decision_point: str = 'T0') -> dict:
    """计算DAI_C/DAI_D/PAVR/PUR. decision_point='T1'时, T1_available/T1_intra变量视为可得."""
    d = registry.copy()
    if dataset:
        d = d[d['dataset'] == dataset]
    elig = d[d['causal_role'].isin(ELIGIBLE_ROLES)]
    if len(elig) == 0:
        raise ValueError('无合格候选预测变量')
    if decision_point == 'T1':
        # 决策特异性: T0的"决策后"变量在T1已产生并进入病历 (假设: 结果已返回)
        T = elig['timing_class'].isin(['T0_available', 'T1_available', 'T1_intra',
                                       'post_decision']).astype(int)
        T = T.where(~elig['timing_class'].eq('timing_unclear'), 'U')
        C_eff = elig['C_clinical'].where(elig['timing_class'].ne('post_decision'), 1)
    else:
        T = elig['T_t0']
        C_eff = elig['C_clinical']
    T1mask = (T == 1)
    idx = {
        'n_total': int(len(d)), 'n_eligible': int(len(elig)),
        'DAI_C': round(float((T1mask & (C_eff == 1)).sum()) / len(elig), 3),
        'DAI_D': round(float((T1mask & (elig['R_shared'] == 1)).sum()) / len(elig), 3),
        'n_T1_R1_C1': int((T1mask & (elig['R_shared'] == 1) & (C_eff == 1)).sum()),
        'n_T1_R0': int((T1mask & (elig['R_shared'] == 0)).sum()),
        'n_T0_R1': int(((T == 0) & (elig['R_shared'] == 1)).sum()),
        'n_timing_unclear': int((T == 'U').sum()),
    }
    model = d[d['in_model'] == 1]
    if len(model):
        idx['n_model_predictors'] = int(len(model))
        idx['PAVR'] = round(float(model['causal_role'].eq('post_decision_predictor').sum()) / len(model), 3)
        idx['PUR'] = round(float(model['causal_role'].eq('candidate_timing_unclear').sum()) / len(model), 3)
    return idx


def detect_issues(registry: pd.DataFrame) -> list[dict]:
    """自动检出审计问题: 缺列/时点不明/决策后依赖/无患者键."""
    issues = []
    unshared = registry[(registry['R_shared'] == 0) & (registry['T_t0'] == 1)]
    for _, r in unshared.iterrows():
        issues.append({'type': 'S3_measured_not_shared', 'variable': r['raw_column'],
                       'detail': f'{r["raw_column"]}: 决策前已测量但共享数据缺列 (T=1,R=0)'})
    post_in_model = registry[(registry['causal_role'] == 'post_decision_predictor') & (registry['in_model'] == 1)]
    if len(post_in_model):
        n_model = int(registry['in_model'].eq(1).sum())
        issues.append({'type': 'S1_post_decision_in_model', 'variable': ';'.join(post_in_model['raw_column']),
                       'detail': f'模型含{len(post_in_model)}个决策后变量 (模型协变量共{n_model}个)'})
    mediator = registry[(registry['causal_role'] == 'mediator') & (registry['in_model'] == 1)]
    if len(mediator):
        issues.append({'type': 'S2_mediator_for_total_effect', 'variable': ';'.join(mediator['raw_column']),
                       'detail': '中介变量用于总效应估计, 条件关联不等于总效应'})
    unclear = registry[(registry['causal_role'] == 'candidate_timing_unclear') & (registry['in_model'] == 1)]
    if len(unclear):
        issues.append({'type': 'timing_unclear_in_model', 'variable': ';'.join(unclear['raw_column']),
                       'detail': f'模型含{len(unclear)}个时点不明变量 (PUR)'})
    if 'patient_key' in registry.columns and (registry['patient_key'] == 0).any():
        issues.append({'type': 'S4_patient_key_missing', 'variable': '-',
                       'detail': '存在患者内聚类但患者键未共享, 聚类校正不可行'})
    return issues


def audit_report(registry: pd.DataFrame, datasets: list[str] | None = None) -> str:
    """生成Markdown审计报告."""
    lines = ['# DTAA审计报告', '', '生成: dtaa_audit | 三层判定 T(时间可得)/R(共享状态)/C(临床可及)', '']
    dss = datasets or list(registry['dataset'].unique())
    for ds in dss:
        lines.append(f'## {ds}')
        for dp in ['T0', 'T1']:
            try:
                idx = compute_indices(registry, ds, dp)
            except ValueError:
                continue
            lines.append(f'- 决策点{dp}: DAI_C={idx["DAI_C"]}, DAI_D={idx["DAI_D"]}, '
                         f'合格变量{idx["n_eligible"]}/{idx["n_total"]}')
        if 'PAVR' in idx:
            lines.append(f'- 模型依赖: PAVR={idx["PAVR"]}, PUR={idx["PUR"]} (模型协变量{idx["n_model_predictors"]}个)')
        for iss in detect_issues(registry[registry['dataset'] == ds]):
            lines.append(f'- **[{iss["type"]}]** {iss["detail"]}')
        lines.append('')
    return '\n'.join(lines)
