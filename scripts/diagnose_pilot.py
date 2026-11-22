"""Diagnose completed development runs without fitting or changing any model."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import pandas as pd
from f4.backtest import _prepare_inputs, _safe_id
from f4.common import atomic_write_json, project_path, sha256_file, source_snapshot, timestamp
from f4.diagnostics import cohort_diagnostics, prediction_diagnostics
from f4.metrics import validate_predictions
from f4.models import load_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot-report', required=True)
    parser.add_argument('--report-id', required=True)
    args = parser.parse_args()
    output = ROOT / 'reports/diagnostics' / _safe_id(args.report_id, 'report-id')
    if output.exists():
        raise FileExistsError('Diagnostic reports must not be overwritten')
    pilot_path = project_path(args.pilot_report)
    pilot = json.loads(pilot_path.read_text(encoding='utf-8'))
    if pilot.get('status') != 'complete' or pilot.get('synthetic') or pilot.get('fold') != 'dev_1':
        raise ValueError('This diagnostic entry point only accepts the completed real dev_1 pilot')
    manifests = []
    for entry in pilot['runs']:
        path = project_path(entry['manifest'])
        if sha256_file(path) != entry['manifest_sha256']:
            raise ValueError('Run manifest has changed')
        manifest = json.loads(path.read_text(encoding='utf-8'))
        if manifest['status'] != 'complete' or manifest['synthetic'] or manifest['fold'] != 'dev_1':
            raise ValueError('Unexpected run state or scope')
        manifests.append((entry, manifest))
    if not manifests:
        raise ValueError('No completed runs to diagnose')
    source, cohorts, splits = _prepare_inputs(manifests[0][1]['config'], 'dev_1', False)
    if any(manifest['source'] != source for _, manifest in manifests):
        raise ValueError('Compared runs must use identical, intact source and frozen cohorts')
    output.mkdir(parents=True, exist_ok=False)
    snapshot_files, snapshot_hash = source_snapshot()
    atomic_write_json(output / 'code_snapshot.json', snapshot_files)
    report = {'created_at': timestamp(), 'status': 'running', 'mode': 'read_only_posthoc_diagnostics',
              'new_models_fitted': False, 'final_holdout_evaluated': False,
              'pilot_report_sha256': sha256_file(pilot_path), 'source': source,
              'diagnostic_code_snapshot_sha256': snapshot_hash,
              'cohorts': cohort_diagnostics(cohorts, source['times'], splits['folds']['dev_1'].get('calibration_maturity')),
              'runs': [], 'limitations': [
                  'Observed score differences compare variants, not old versus new code versions',
                  'No calibrated carrier baseline real result exists yet; calibration and model contributions are not isolated',
                  'Maturity is measured within eventually-signed official records, not all arriving parcels',
                  'Descriptive diagnostics cannot identify a unique causal explanation',
                  'Original pilot memory counters measured venv launchers and are invalid for training memory']}
    report_path = output / 'diagnostics.json'
    atomic_write_json(report_path, report)
    try:
        evaluation = cohorts['eval'][['pkg_no', 'lgs_company_nm', 'y_total_hours']]
        for entry, manifest in manifests:
            prediction_path = project_path(manifest['predictions']['path'])
            if sha256_file(prediction_path) != manifest['predictions']['sha256']:
                raise ValueError('Predictions have changed')
            predictions = pd.read_csv(prediction_path, dtype={'pkg_no': 'string'})
            diagnostics = prediction_diagnostics(evaluation, predictions)
            if any(abs(diagnostics['metrics'][key] - manifest['official_metrics'][key]) >= 1e-8
                   for key in diagnostics['metrics']):
                raise ValueError('Recomputed metrics disagree with recorded official results')
            detail = {'run_id': manifest['run_id'], 'experiment': manifest['experiment'],
                      'manifest_sha256': entry['manifest_sha256'],
                      'elapsed_seconds': entry['elapsed_seconds'], 'evaluation': diagnostics}
            if 'calibrator' in manifest['artifacts']:
                cal_path = project_path(manifest['locations']['outputs']) / 'calibration_base_predictions.csv'
                if sha256_file(cal_path) != manifest['calibration_base_predictions_sha256']:
                    raise ValueError('Calibration base predictions have changed')
                artifact = manifest['artifacts']['calibrator']
                artifact_path = project_path(artifact['path']).resolve()
                if not artifact_path.is_relative_to((ROOT / 'artifacts/runs').resolve()):
                    raise ValueError('Only trusted project run artifacts may be loaded')
                if sha256_file(artifact_path) != artifact['sha256']:
                    raise ValueError('Calibrator artifact has changed')
                calibrator = load_model(artifact_path)
                if (calibrator.fit_ids_sha256_ != manifest['model_fit_ids_sha256'] or
                        calibrator.cal_ids_sha256_ != manifest['calibration_ids_sha256']):
                    raise ValueError('Calibrator cohort binding changed')
                cal_truth = cohorts['cal'][['pkg_no', 'lgs_company_nm', 'y_total_hours']]
                cal_base = pd.read_csv(cal_path, dtype={'pkg_no': 'string'})
                aligned = validate_predictions(cal_truth, cal_base)
                adjusted = calibrator.apply(aligned.pred_hours, aligned.lgs_company_nm)
                detail['calibration'] = {
                    'scope': 'Resubstitution on calibration data, not independent validation',
                    'before': prediction_diagnostics(cal_truth, cal_base),
                    'after': prediction_diagnostics(cal_truth, pd.DataFrame({'pkg_no': aligned.pkg_no, 'pred_hours': adjusted})),
                    'carrier_offsets_hours': calibrator.group_deltas_}
            report['runs'].append(detail)
            atomic_write_json(report_path, report)
        report.update(status='complete', finished_at=timestamp())
        atomic_write_json(report_path, report)
        print(f'Diagnostics complete: {report_path}')
    except BaseException as exc:
        report.update(status='failed', finished_at=timestamp(), error=f'{type(exc).__name__}: {exc}')
        atomic_write_json(report_path, report)
        raise


if __name__ == '__main__':
    main()
