"""Audit frozen development calibration tails, excluding holdout before labels."""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import yaml

from f4.backtest import _safe_id
from f4.common import atomic_write_json, sha256_file, source_snapshot, timestamp
from f4.features import pkg_ids_sha256, validate_pkg_ids
from f4.platform_utils import current_process_peak_memory
from f4.splits import _key_digest, validate_config
from f4.tail_audit import audit_tail_queue


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-id', required=True)
    args = parser.parse_args()
    output = ROOT / 'reports/diagnostics' / _safe_id(args.report_id, 'report-id')
    if output.exists():
        raise FileExistsError('Audit IDs cannot be reused')
    started = time.perf_counter()
    core = ROOT / 'data/processed/core/train.parquet'
    config_path = ROOT / 'configs/splits.yaml'
    split_id, holdout_spec, development = validate_config(yaml.safe_load(config_path.read_text(encoding='utf-8')))
    report_path = ROOT / 'reports/splits' / (split_id + '.json')
    split = json.loads(report_path.read_text(encoding='utf-8'))
    canonical = {'split_id': split_id, 'holdout': holdout_spec.metadata(),
                 'development': [s.metadata() for s in development]}
    if (split['status'] != 'complete' or not split['holdout_isolation_checked']
            or split['config'] != canonical
            or split['config_file_sha256'] != sha256_file(config_path)
            or split['train_parquet_sha256'] != sha256_file(core)):
        raise ValueError('Frozen split source mismatch')

    def keys(meta):
        path = Path(meta['path'])
        if sha256_file(path) != meta['file_sha256']:
            raise ValueError('Key artifact changed')
        frame = pd.read_parquet(path, columns=['pkg_no'], use_threads=False)
        validate_pkg_ids(frame.pkg_no, len(frame))
        if len(frame) != meta['n'] or _key_digest(frame) != meta['keys_sha256']:
            raise ValueError('Key content changed')
        return frame

    holdout = keys(split['holdout'])
    dataset = ds.dataset(core, format='parquet')
    issue_type = dataset.schema.field('issue_time').type
    files, snapshot = source_snapshot()
    output.mkdir(parents=True, exist_ok=False)
    atomic_write_json(output / 'code_snapshot.json', files)
    result = {'status': 'running', 'started_at': timestamp(), 'mode': 'posthoc_development_tail_audit',
              'new_models_fitted': False, 'new_predictions_created': False,
              'final_holdout_evaluated': False, 'final_holdout_labels_materialized': False,
              'split_report_sha256': sha256_file(report_path),
              'core_sha256': split['train_parquet_sha256'], 'source_snapshot_sha256': snapshot,
              'folds': {}}
    target = output / 'tail_audit.json'
    atomic_write_json(target, result)
    try:
        for name in ('dev_1', 'dev_2'):
            fold = split['folds'][name]
            times = fold['times']
            if times != next(s.metadata() for s in development if s.name == name):
                raise ValueError('Fold time mismatch')
            cal = keys(fold['artifacts']['cal'])
            evaluation_keys = keys(fold['artifacts']['eval'])
            # Read only keys for the full calibration issue queue first.
            condition = ((ds.field('issue_time') >= pa.scalar(pd.Timestamp(times['F']), type=issue_type))
                         & (ds.field('issue_time') < pa.scalar(pd.Timestamp(times['E']), type=issue_type)))
            queue_keys = dataset.to_table(columns=['pkg_no'], filter=condition, use_threads=False).to_pandas()
            validate_pkg_ids(queue_keys.pkg_no, len(queue_keys))
            selected = pd.concat([queue_keys, evaluation_keys], ignore_index=True)
            validate_pkg_ids(selected.pkg_no, len(selected))
            if selected.pkg_no.isin(holdout.pkg_no).any():
                raise ValueError('Development audit would touch holdout labels')
            # Arrow filters to approved development keys before labels enter pandas.
            columns = ['pkg_no', 'lgs_company_nm', 'issue_time', 'pay_time', 'sign_time', 'y_total_hours']
            source = dataset.to_table(columns=columns, filter=ds.field('pkg_no').isin(selected.pkg_no.tolist()),
                                      use_threads=False).to_pandas()
            validate_pkg_ids(source.pkg_no, len(source))

            def aligned(wanted):
                joined = wanted.merge(source, on='pkg_no', how='left', sort=False, validate='one_to_one', indicator=True)
                if (len(joined) != len(wanted) or not joined['_merge'].eq('both').all()
                        or joined.pkg_no.tolist() != wanted.pkg_no.tolist()):
                    raise ValueError('Diagnostic join incomplete or changed order')
                return joined.drop(columns='_merge')

            queue, evaluation = aligned(queue_keys), aligned(evaluation_keys)
            visible_ids = queue.loc[pd.to_datetime(queue.sign_time).lt(pd.Timestamp(times['C'])), 'pkg_no']
            if len(visible_ids) != len(cal) or set(visible_ids) != set(cal.pkg_no):
                raise ValueError('Reconstructed available queue differs from frozen calibration keys')
            audit = audit_tail_queue(queue, evaluation, times)
            for row in audit['by_carrier']:
                recorded = next(x for x in fold['calibration_maturity']['by_carrier'] if x['lgs_company_nm'] == row['group'])
                if recorded['queue_n'] != row['queue_n'] or recorded['mature_n'] != row['available_n']:
                    raise ValueError('Queue maturity differs from frozen split audit')
            audit.update(times=times, queue_keys_sha256=pkg_ids_sha256(queue_keys.pkg_no),
                         queue_n=len(queue), evaluation_keys=fold['artifacts']['eval'],
                         available_keys_match_frozen_calibration=True, holdout_key_intersection=0)
            result['folds'][name] = audit
            atomic_write_json(target, result)
            print(f'Completed {name} descriptive tail audit', flush=True)
            del source, queue, evaluation, selected
        if source_snapshot()[1] != snapshot:
            raise ValueError('Source changed during audit')
        result.update(status='complete', finished_at=timestamp(),
                      elapsed_seconds=round(time.perf_counter() - started, 3),
                      resources=current_process_peak_memory())
        atomic_write_json(target, result)
        print(target)
    except BaseException as exc:
        result.update(status='failed', finished_at=timestamp(), error=f'{type(exc).__name__}: {exc}')
        atomic_write_json(target, result)
        raise


if __name__ == '__main__':
    main()
