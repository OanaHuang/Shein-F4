"""Compare completed deployment-C survival model runs with their C-M1 references."""

import argparse
import json
import math
from pathlib import Path
import sys

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from f4.common import sha256_file
from f4.features import validate_pkg_ids
from f4.metrics import prediction_diagnostics, read_prediction_csv, score_predictions
from f4.splits import DEPLOY_C_PROTOCOL, _identifier, _key_digest

DEVELOPMENT_FOLDS = ('dev_1', 'dev_2')
MODEL_NAMES = ('C-M1', 'AFT exact-only', 'AFT right-censored')
SURVIVAL_NAMES = {'AFT': MODEL_NAMES[1:], 'NGBOOST': ('NGBoost exact-only', 'NGBoost right-censored')}
COHORT_FIELDS = ('n', 'keys_sha256', 'file_sha256')
METRICS = ('MAE', 'mean_pred_hours', 'severe_rate')


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def verify_file(path, digest):
    if sha256_file(path) != digest:
        raise ValueError(f'File changed since the frozen run: {path}')
    return path


def cohort_identity(record):
    return tuple(record[field] for field in COHORT_FIELDS)


def read_keys(path, record, *, allow_empty=False):
    frame = pd.read_parquet(verify_file(path, record['file_sha256']),
                            columns=['pkg_no'], use_threads=False)
    if not (allow_empty and frame.empty and record['n'] == 0):
        validate_pkg_ids(frame['pkg_no'], record['n'])
    if _key_digest(frame) != record['keys_sha256']:
        raise ValueError(f'Frozen key contents changed: {path}')
    return frame


def plan_arms(plan, root, experiment):
    names = SURVIVAL_NAMES[experiment]
    planned = {}
    for job in plan['jobs']:
        run_id = _identifier(job['run_id'], 'run ID')
        config = yaml.safe_load((root / job['config']).read_text(encoding='utf-8'))
        include_censored = config.get(experiment.lower(), {}).get('include_right_censored')
        if config.get('experiment') != experiment or type(include_censored) is not bool:
            raise ValueError(f'Expected a deployment-C {experiment} configuration: {job["config"]}')
        key = job['fold'], names[1 if include_censored else 0]
        if key in planned:
            raise ValueError(f'Duplicate {experiment} comparison entry: {key}')
        planned[key] = (run_id, experiment, include_censored)
    if set(planned) != {(fold, name) for fold in DEVELOPMENT_FOLDS for name in names}:
        raise ValueError(f'Both {experiment} variants must be planned once per development window')
    return planned


def load_runs(plan_path, root):
    plan = yaml.safe_load((root / plan_path).read_text(encoding='utf-8'))
    aft_plan = plan
    names = MODEL_NAMES
    aft_reference_plan = plan.get('aft_reference_plan')
    if aft_reference_plan is not None:
        aft_plan = yaml.safe_load((root / aft_reference_plan).read_text(encoding='utf-8'))
        names += SURVIVAL_NAMES['NGBOOST']
    baseline_runs = aft_plan['baseline_runs']
    if set(baseline_runs) != set(DEVELOPMENT_FOLDS):
        raise ValueError('The comparison requires one C-M1 reference per development window')
    planned = plan_arms(aft_plan, root, 'AFT')
    if aft_reference_plan is not None:
        planned.update(plan_arms(plan, root, 'NGBOOST'))
    for fold, run_id in baseline_runs.items():
        planned[fold, 'C-M1'] = (_identifier(run_id, 'baseline run ID'), 'M1', None)
    if len({run_id for run_id, _, _ in planned.values()}) != len(planned):
        raise ValueError('Each comparison arm requires a unique run ID')

    runs = {}
    scorer_hash = sha256_file(root / 'official/tools/score.py')
    for fold in DEVELOPMENT_FOLDS:
        runs[fold] = {}
        for name in names:
            run_id, experiment, include_censored = planned[fold, name]
            manifest_path = root / 'reports/runs' / run_id / 'run_manifest.json'
            if not manifest_path.exists():
                raise ValueError(f'Comparison requires a completed run: {run_id}')
            manifest = read_json(manifest_path)
            if (manifest.get('status') != 'complete' or manifest.get('synthetic') is not False
                    or manifest.get('run_id') != run_id or manifest.get('fold') != fold
                    or manifest.get('protocol') != DEPLOY_C_PROTOCOL
                    or manifest.get('experiment') != experiment
                    or manifest.get('official_scorer_sha256') != scorer_hash):
                raise ValueError(f'Invalid completed deployment-C result: {run_id}')
            if include_censored is not None and (
                    manifest['config'].get(experiment.lower(), {}).get('include_right_censored')
                    is not include_censored):
                raise ValueError(f'{experiment} run does not match its comparison arm: {run_id}')
            runs[fold][name] = manifest
    return runs


def baseline_diagnostics(train_path, frozen, prediction_path, manifest):
    keys = frozen['pkg_no']
    truth = pd.read_parquet(train_path, columns=['pkg_no', 'lgs_company_nm', 'y_total_hours'],
                            filters=[('pkg_no', 'in', keys.tolist())], use_threads=False)
    truth = frozen.merge(truth, on='pkg_no', how='left', sort=False,
                         validate='one_to_one', indicator=True)
    if not truth['_merge'].eq('both').all():
        raise ValueError('Evaluation source does not contain every frozen key')
    predictions = read_prediction_csv(prediction_path)
    if not predictions['pkg_no'].equals(keys.astype('string')):
        raise ValueError('Saved C-M1 prediction order differs from frozen evaluation keys')
    score, _ = score_predictions(truth, predictions)
    if any(not math.isclose(value, manifest['metrics'][key], rel_tol=0, abs_tol=1e-8)
           for key, value in score.items()):
        raise ValueError('Saved C-M1 scores differ from its verified predictions')
    return prediction_diagnostics(truth, predictions)


def verify_survival_pairs(models, fold):
    baseline_features = models['C-M1']['config'].get('features')
    encoder_hash = models[MODEL_NAMES[1]]['aft_training']['encoder_fit_ids_sha256']
    for experiment, (exact_name, censored_name) in SURVIVAL_NAMES.items():
        if exact_name not in models:
            continue
        exact, censored = models[exact_name], models[censored_name]
        if (exact['config'].get('model') != censored['config'].get('model')
                or any(arm['config'].get('features') != baseline_features for arm in (exact, censored))):
            raise ValueError(f'{experiment} variants must share model parameters and C-M1 features: {fold}')
        if any(arm[f'{experiment.lower()}_training']['encoder_fit_ids_sha256'] != encoder_hash
               for arm in (exact, censored)):
            raise ValueError(f'Survival variants must share the frozen encoder cohort: {fold}')


def compare(plan_path, root=ROOT):
    runs = load_runs(plan_path, root)
    source = runs['dev_1']['C-M1']['source']
    split_id = _identifier(source['split_id'], 'split ID')
    split_report = read_json(verify_file(root / 'reports/splits' / f'{split_id}.json',
                                         source['split_report_sha256']))
    train_path = verify_file(root / 'data/processed/core/train.parquet',
                             source['train_parquet_sha256'])
    if (split_report.get('status') != 'complete'
            or split_report.get('holdout_isolation_checked') is not True
            or split_report.get('split_id') != split_id
            or split_report['train_parquet_sha256'] != source['train_parquet_sha256']
            or split_report['config_sha256'] != source['split_config_sha256']):
        raise ValueError('Comparison requires the original isolated development split')
    split_root = root / 'artifacts/splits' / split_id
    holdout = read_keys(split_root / 'holdout_ids.parquet', split_report['holdout'])

    rows = []
    for fold, models in runs.items():
        frozen_fold = split_report['folds'][fold]
        verify_survival_pairs(models, fold)
        frozen_eval = read_keys(split_root / fold / 'eval_ids.parquet',
                                frozen_fold['artifacts']['eval'])
        if frozen_eval['pkg_no'].isin(holdout['pkg_no']).any():
            raise ValueError('Development evaluation keys overlap the locked holdout')
        censor_identity = None
        for name, manifest in models.items():
            run_id = manifest['run_id']
            run_source = manifest['source']
            cohorts = run_source['cohort_keys']
            if (any(run_source[field] != source[field] for field in
                    ('split_id', 'split_report_sha256', 'train_parquet_sha256', 'split_config_sha256'))
                    or run_source['times'] != frozen_fold['times']
                    or any(cohort_identity(cohorts[phase]) != cohort_identity(frozen_fold['artifacts'][phase])
                           for phase in ('fit', 'cal', 'eval'))):
                raise ValueError(f'Run used different data or frozen cohorts: {run_id}')

            right_censored_n = 0
            if name != 'C-M1':
                experiment = manifest['experiment'].lower()
                training = manifest[f'{experiment}_training']
                include_censored = manifest['config'][experiment]['include_right_censored']
                right_censored_n = training['right_censored_n']
                record = cohorts.get(f'{experiment}_censored_fit')
                if (training['exact_n'] != cohorts['fit']['n']
                        or training['training_n'] != training['exact_n'] + right_censored_n
                        or (not include_censored and (right_censored_n or record is not None))
                        or (include_censored and (record is None or right_censored_n != record['n']))):
                    raise ValueError(f'Survival training counts do not match the comparison arm: {run_id}')
                if record is not None:
                    censored = read_keys(root / record['path'], record, allow_empty=True)
                    if censored['pkg_no'].isin(holdout['pkg_no']).any():
                        raise ValueError(f'Censored fit keys overlap the locked holdout: {run_id}')
                    current_identity = (record['n'], record['keys_sha256'])
                    if censor_identity is not None and current_identity != censor_identity:
                        raise ValueError(f'Survival variants used different right-censored fit keys: {fold}')
                    censor_identity = current_identity

            prediction = manifest['predictions']
            prediction_path = verify_file(root / prediction['path'], prediction['sha256'])
            diagnostics = manifest.get('evaluation_diagnostics')
            if diagnostics is None and name == 'C-M1':
                diagnostics = baseline_diagnostics(train_path, frozen_eval, prediction_path, manifest)
            official_a = manifest['official_metrics']['A']
            if (diagnostics is None or not all(math.isfinite(diagnostics[key]) for key in METRICS)
                    or diagnostics['MAE'] < 0 or diagnostics['mean_pred_hours'] <= 0
                    or not 0 <= diagnostics['severe_rate'] <= 1
                    or not math.isclose(official_a, manifest['metrics']['A'], rel_tol=0, abs_tol=1e-8)):
                raise ValueError(f'Invalid evaluation metrics: {run_id}')
            rows.append({'fold': fold, 'C': frozen_fold['times']['C'], 'model': name,
                         'run_id': run_id, 'eval_n': len(frozen_eval),
                         'right_censored_fit_n': right_censored_n,
                         'A': official_a, **{key: diagnostics[key] for key in METRICS}})
    return {'protocol': DEPLOY_C_PROTOCOL, 'split_id': split_id, 'results': rows}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', default='configs/training/deploy_C_aft_comparison.yaml')
    parser.add_argument('--json', action='store_true', help='Print machine-readable results')
    args = parser.parse_args(argv)
    try:
        result = compare(args.plan)
    except (FileNotFoundError, ValueError) as error:
        parser.exit(2, f'{error}\n')
    if args.json:
        print(json.dumps(result, indent=2, allow_nan=False))
    else:
        print('| Window | Model | A | MAE (h) | Mean commitment (h) | Severe late | Right-censored fit n |')
        print('|---|---|---:|---:|---:|---:|---:|')
        for row in result['results']:
            print(f"| {row['fold']} | {row['model']} | {row['A']:.4f} | {row['MAE']:.2f} | "
                  f"{row['mean_pred_hours']:.2f} | {row['severe_rate']:.2%} | "
                  f"{row['right_censored_fit_n']} |")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
