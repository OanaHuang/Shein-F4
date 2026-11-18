"""Bounded-memory preparation and audits; these functions do not fit models."""

from __future__ import annotations

from collections import Counter
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SHARED_COLUMNS = [
    'pkg_no', 'pkg_goods_cnt', 'weight', 'shpp_country_nm', 'shpp_prvn_nm',
    'shpp_city_nm', 'stk_cate_tp_nm', 'shpp_chnl', 'lgs_company_nm',
    'combine_flag', 'combine_type', 'start_airport_cd', 'end_airport_cd',
    'mode_category', 'pay_mthd', 'business_tp', 'operation_type', 'route_cat',
    'sale_area', 'send_country_nm', 'shpp_post_no', 'pay_time', 'issue_time',
]
EVENT_COLUMNS = [
    'occupy_time', 'pack_time', 'preshpp_time', 'closebox_time', 'start_time',
    'task_finish_time', 'sending_time', 'goods_send_time', 'combine_pkg_send_time',
    'fillcarshpp_time', 'fillcarcmplt_time', 'plantakeoff_time', 'takeoff_time',
    'arrived_unload_place_time', 'tran_arrival_time', 'customs_clear_time',
    'outbound_transfer_time', 'fully_released_time', 'arrive_time', 'noa',
    'arriving_gha', 'stocked', 'out_of_stock', 'waiting_for_pick_up',
    'arriving_lm_hub', 'fm_arrive_time', 'fm_collect_time', 'accept_time',
    'outsend_time', 'shpp_time', 'merchant_shpp_time', 'subfillcar_time',
    'mainreceive_time', 'subpack_time', 'subclosebox_time', 'subarrive_time',
    'status_7_time', 'status_8_time', 'wait_for_collect_time', 'collectsign_time',
    'outsendcollectsign_time', 'sub_transport_time_parsed', 'sign_time',
]
TRAIN_COLUMNS = SHARED_COLUMNS + EVENT_COLUMNS + ['y_total_hours']
TRAIN_CORE_COLUMNS = SHARED_COLUMNS + ['sign_time', 'y_total_hours']
NUMERIC_COLUMNS = ['pkg_goods_cnt', 'weight']
TIME_COLUMNS = ['pay_time', 'issue_time']
CATEGORICAL_COLUMNS = [c for c in SHARED_COLUMNS
                       if c not in {'pkg_no', 'shpp_post_no', *NUMERIC_COLUMNS, *TIME_COLUMNS}]
EXPECTED_ROWS = {'train': 5_267_074, 'test': 911_415}
FILENAMES = {'train': 'train_202608_0920.csv', 'test': 'test_202609.csv'}


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def _sha256(path: Path) -> str:
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def _timestamp() -> str:
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')


def _schema(train: bool) -> pa.Schema:
    columns = TRAIN_CORE_COLUMNS if train else SHARED_COLUMNS
    fields = []
    for column in columns:
        if column in TIME_COLUMNS + ['sign_time']:
            dtype = pa.timestamp('ns')
        elif column in NUMERIC_COLUMNS + ['y_total_hours']:
            dtype = pa.float64()
        else:
            dtype = pa.string()
        fields.append(pa.field(column, dtype))
    return pa.schema(fields)


def _normalize(frame: pd.DataFrame, train: bool) -> tuple[pd.DataFrame, dict]:
    anomalies = Counter()
    for column in TIME_COLUMNS + (['sign_time'] if train else []):
        original = frame[column]
        parsed = pd.to_datetime(original, format='%Y-%m-%d %H:%M:%S', errors='coerce')
        anomalies[f'{column}_invalid_parse'] += int((original.notna() & parsed.isna()).sum())
        frame[column] = parsed
    for column in NUMERIC_COLUMNS + (['y_total_hours'] if train else []):
        original = frame[column]
        converted = pd.to_numeric(original, errors='coerce').astype('float64')
        invalid = original.notna() & (~np.isfinite(converted))
        anomalies[f'{column}_invalid_numeric'] += int(invalid.sum())
        frame[column] = converted.where(np.isfinite(converted), np.nan)
    anomalies['negative_issue_minus_pay'] += int((frame['issue_time'] < frame['pay_time']).sum())
    anomalies['nonpositive_weight'] += int((frame['weight'] <= 0).sum())
    anomalies['nonpositive_goods_count'] += int((frame['pkg_goods_cnt'] <= 0).sum())
    goods = frame['pkg_goods_cnt']
    anomalies['noninteger_goods_count'] += int((goods.notna() & (goods != np.floor(goods))).sum())
    if train:
        anomalies['label_outside_0_to_2160_hours'] += int(
            ((frame['y_total_hours'] <= 0) | (frame['y_total_hours'] >= 2160)).sum())
    return frame, dict(anomalies)


def prepare_data(raw_root: str | Path, out: str | Path, *, chunk_size: int = 200_000,
                 expected_rows: dict | None = None, overwrite: bool = False) -> dict:
    """Create canonical 25/23-column Parquet tables, retaining every input row.

    Invalid numeric/time text is represented as null and counted. Its original
    representation stays in the unchanged CSV, identified by a complete SHA-256.
    """
    raw_root, out = Path(raw_root).resolve(), Path(out).resolve()
    expected_rows = EXPECTED_ROWS if expected_rows is None else expected_rows
    if chunk_size < 1:
        raise ValueError('chunk_size must be positive')
    if out == raw_root or raw_root.is_relative_to(out) or out.is_relative_to(raw_root):
        raise ValueError('Processed output must not replace or contain the raw root')
    out.mkdir(parents=True, exist_ok=True)
    if not overwrite and any((out / f'{split}.parquet').exists() for split in FILENAMES):
        raise FileExistsError('Core cache already exists; inspect provenance or use --overwrite')
    manifest = {'created_at': _timestamp(), 'schema_version': 1, 'status': 'preparing',
                'rows_removed': 0, 'chunk_size': chunk_size, 'files': {},
                'protocol': 'preparation_only_no_model_fit'}
    temporary_files = []
    with tempfile.TemporaryDirectory(prefix='f4-key-check-', dir=out) as temporary:
        db = sqlite3.connect(str(Path(temporary) / 'keys.sqlite'))
        db.execute('PRAGMA journal_mode=OFF')
        db.execute('PRAGMA synchronous=OFF')
        db.execute('PRAGMA cache_size=-32768')
        db.execute('CREATE TABLE keys (pkg_no TEXT PRIMARY KEY, split TEXT NOT NULL) WITHOUT ROWID')
        try:
            for split, filename in FILENAMES.items():
                matches = list(raw_root.rglob(filename))
                if len(matches) != 1:
                    raise ValueError(f'Expected one {filename}, found {len(matches)}')
                source = matches[0]
                with source.open(newline='', encoding='utf-8-sig') as stream:
                    header = next(csv.reader(stream))
                expected_columns = TRAIN_COLUMNS if split == 'train' else SHARED_COLUMNS
                if len(set(header)) != len(header) or set(header) != set(expected_columns):
                    raise ValueError(f'{split} schema mismatch: expected {len(expected_columns)} official columns')
                source_hash = _sha256(source)
                schema = _schema(split == 'train').with_metadata({
                    b'f4.schema_version': b'1', b'f4.raw_sha256': source_hash.encode(),
                    b'f4.split': split.encode(),
                })
                target = out / f'{split}.parquet.partial'
                temporary_files.append(target)
                count = 0
                anomalies = Counter()
                core_columns = TRAIN_CORE_COLUMNS if split == 'train' else SHARED_COLUMNS
                reader = pd.read_csv(source, usecols=core_columns, dtype='string',
                                     chunksize=chunk_size, keep_default_na=False,
                                     na_values=[''], encoding='utf-8-sig')
                with reader, pq.ParquetWriter(target, schema, compression='zstd') as writer:
                    for frame in reader:
                        frame = frame[core_columns]
                        if frame['pkg_no'].isna().any() or frame['pkg_no'].eq('').any():
                            raise ValueError(f'{split} has empty package keys')
                        keys = frame['pkg_no'].tolist()
                        if len(set(keys)) != len(keys):
                            raise ValueError(f'{split} has duplicate package keys within a chunk')
                        try:
                            with db:
                                db.executemany('INSERT INTO keys VALUES (?, ?)',
                                               ((key, split) for key in keys))
                        except sqlite3.IntegrityError as exc:
                            raise ValueError(f'{split} package keys overlap an earlier chunk or dataset') from exc
                        frame, issues = _normalize(frame, split == 'train')
                        anomalies.update(issues)
                        writer.write_table(pa.Table.from_pandas(frame, schema=schema, preserve_index=False))
                        count += len(frame)
                expected = expected_rows.get(split)
                if expected is not None and count != expected:
                    raise ValueError(f'{split} rows {count} != expected {expected}')
                manifest['files'][split] = {
                    'source_path': str(source), 'source_bytes': source.stat().st_size,
                    'source_sha256': source_hash, 'rows': count,
                    'raw_columns': len(header), 'columns': core_columns,
                    'parquet_path': f'{split}.parquet', 'parquet_bytes': target.stat().st_size,
                    'parquet_sha256': _sha256(target),
                    'keys_unique': True, 'normalization_anomalies': dict(anomalies),
                }
                print(f'{split}: prepared {count:,} rows', flush=True)
            manifest.update(status='complete', keys_unique_across_train_test=True)
            for split in FILENAMES:
                (out / f'{split}.parquet.partial').replace(out / f'{split}.parquet')
            _write_json(out / 'preparation_manifest.json', manifest)
            _write_json(out / 'schema.json', {
                'schema_version': 1, 'shared_columns': SHARED_COLUMNS,
                'train_core_columns': TRAIN_CORE_COLUMNS,
                'train_types': {field.name: str(field.type) for field in _schema(True)},
                'excluded_per_package_event_columns': EVENT_COLUMNS,
                'normalization': 'empty text to null; invalid numeric/time to null, counted; raw CSV unchanged',
            })
        except Exception as exc:
            manifest.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            _write_json(out / 'preparation_failure.json', manifest)
            for path in temporary_files:
                path.unlink(missing_ok=True)
            raise
        finally:
            db.close()
    return manifest


class _Distinct:
    """Small sets stay in memory; high-cardinality values spill to SQLite."""
    def __init__(self, db: sqlite3.Connection, split: str, column: str, threshold=50_000):
        self.db, self.split, self.column, self.threshold = db, split, column, threshold
        self.values = set()
        self.spilled = False

    def add(self, values) -> None:
        unique = {str(value) for value in values.dropna().drop_duplicates().tolist()}
        if not self.spilled and len(self.values) + len(unique) <= self.threshold:
            self.values.update(unique)
            return
        if not self.spilled:
            unique.update(self.values)
            self.values.clear()
            self.spilled = True
        self.db.executemany('INSERT OR IGNORE INTO vals VALUES (?, ?, ?)',
                            ((self.split, self.column, value) for value in unique))

    def count(self) -> int:
        if not self.spilled:
            return len(self.values)
        return self.db.execute('SELECT COUNT(*) FROM vals WHERE split=? AND col=?',
                               (self.split, self.column)).fetchone()[0]

    def materialize(self) -> None:
        if not self.spilled:
            self.db.executemany('INSERT OR IGNORE INTO vals VALUES (?, ?, ?)',
                                ((self.split, self.column, value) for value in self.values))
            self.values.clear()
            self.spilled = True


def _core_manifest(data_dir: Path) -> dict:
    manifest = json.loads((data_dir / 'preparation_manifest.json').read_text())
    if manifest.get('status') != 'complete' or not manifest.get('keys_unique_across_train_test'):
        raise ValueError('A complete, key-validated preparation manifest is required')
    for split in FILENAMES:
        path = data_dir / f'{split}.parquet'
        if pq.ParquetFile(path).metadata.num_rows != manifest['files'][split]['rows']:
            raise ValueError(f'{split} Parquet row count changed after preparation')
        if _sha256(path) != manifest['files'][split]['parquet_sha256']:
            raise ValueError(f'{split} Parquet hash changed after preparation')
    return manifest


def audit_structure(data_dir: str | Path, out: str | Path, *, batch_size=200_000,
                    distinct_memory_limit=50_000) -> dict:
    data_dir, out = Path(data_dir).resolve(), Path(out).resolve()
    source_manifest = _core_manifest(data_dir)
    report = {'created_at': _timestamp(), 'mode': 'structure',
              'target_labels_inspected': False, 'datasets': {}, 'category_coverage': {},
              'preparation_manifest_sha256': _sha256(data_dir / 'preparation_manifest.json'),
              'raw_source_sha256': {s: f['source_sha256'] for s, f in source_manifest['files'].items()}}
    with tempfile.TemporaryDirectory(prefix='f4-distinct-', dir=data_dir) as temporary:
        db = sqlite3.connect(str(Path(temporary) / 'distinct.sqlite'))
        db.execute('PRAGMA journal_mode=OFF')
        db.execute('PRAGMA synchronous=OFF')
        db.execute('PRAGMA cache_size=-32768')
        db.execute('CREATE TABLE vals(split TEXT, col TEXT, value TEXT, PRIMARY KEY(split,col,value)) WITHOUT ROWID')
        counters = {}
        try:
            for split in FILENAMES:
                nulls, rows = Counter(), 0
                numeric_issues, daily, carrier = Counter(), {c: Counter() for c in TIME_COLUMNS}, Counter()
                ranges = {c: {'min': None, 'max': None} for c in TIME_COLUMNS}
                trackers = {column: _Distinct(db, split, column, threshold=distinct_memory_limit)
                            for column in SHARED_COLUMNS if column != 'pkg_no'}
                counters[split] = trackers
                for batch in pq.ParquetFile(data_dir / f'{split}.parquet').iter_batches(
                        batch_size=batch_size, columns=SHARED_COLUMNS):
                    frame = batch.to_pandas()
                    rows += len(frame)
                    for column in SHARED_COLUMNS:
                        nulls[column] += int(frame[column].isna().sum())
                        if column != 'pkg_no':
                            trackers[column].add(frame[column])
                    for column in TIME_COLUMNS:
                        valid = frame[column].dropna()
                        if len(valid):
                            low, high = valid.min().isoformat(), valid.max().isoformat()
                            ranges[column]['min'] = min(filter(None, [ranges[column]['min'], low]))
                            ranges[column]['max'] = max(filter(None, [ranges[column]['max'], high]))
                            daily[column].update({str(k): int(v) for k, v in valid.dt.strftime('%Y-%m-%d').value_counts().items()})
                    carrier.update({str(k): int(v) for k, v in frame['lgs_company_nm'].dropna().value_counts().items()})
                    numeric_issues['negative_issue_minus_pay'] += int((frame['issue_time'] < frame['pay_time']).sum())
                    numeric_issues['nonpositive_weight'] += int((frame['weight'] <= 0).sum())
                    numeric_issues['nonpositive_goods_count'] += int((frame['pkg_goods_cnt'] <= 0).sum())
                    db.commit()
                fields = {}
                for column in SHARED_COLUMNS:
                    count = rows if column == 'pkg_no' else trackers[column].count()
                    fields[column] = {'missing': nulls[column], 'missing_rate': nulls[column] / rows if rows else 0,
                                      'unique_non_null': count}
                report['datasets'][split] = {'rows': rows, 'core_columns': len(pq.read_schema(data_dir / f'{split}.parquet')),
                    'keys_unique': source_manifest['files'][split]['keys_unique'],
                    'fields': fields, 'time_ranges': ranges,
                    'daily_rows': {c: dict(sorted(v.items())) for c, v in daily.items()},
                    'logistics_rows': dict(sorted(carrier.items())),
                    'anomalies': dict(numeric_issues)}
            for column in CATEGORICAL_COLUMNS + ['shpp_post_no']:
                train, test = counters['train'][column], counters['test'][column]
                if not train.spilled and not test.spilled:
                    unknown = test.values - train.values
                    unknown_count = len(unknown)
                else:
                    train.materialize(); test.materialize(); db.commit()
                    unknown_count = db.execute('SELECT COUNT(*) FROM vals t WHERE t.split="test" AND t.col=? '
                        'AND NOT EXISTS (SELECT 1 FROM vals r WHERE r.split="train" AND r.col=t.col AND r.value=t.value)',
                        (column,)).fetchone()[0]
                    unknown = None
                unknown_rows = 0
                # Membership is checked only for distinct values in each batch, not each SQL row.
                for batch in pq.ParquetFile(data_dir / 'test.parquet').iter_batches(batch_size=batch_size, columns=[column]):
                    counts = batch.to_pandas()[column].dropna().astype(str).value_counts()
                    if unknown is not None:
                        unknown_rows += int(counts[counts.index.isin(unknown)].sum())
                    else:
                        # A temporary join table avoids one SQL request per package.
                        db.execute('CREATE TEMP TABLE IF NOT EXISTS batch_values(value TEXT PRIMARY KEY, n INTEGER) WITHOUT ROWID')
                        db.execute('DELETE FROM batch_values')
                        db.executemany('INSERT INTO batch_values VALUES (?,?)', ((str(k), int(v)) for k,v in counts.items()))
                        unknown_rows += int(db.execute('SELECT COALESCE(SUM(n),0) FROM batch_values b WHERE NOT EXISTS '
                            '(SELECT 1 FROM vals r WHERE r.split="train" AND r.col=? AND r.value=b.value)', (column,)).fetchone()[0])
                report['category_coverage'][column] = {'test_unknown_unique': int(unknown_count),
                    'test_unknown_rows': unknown_rows, 'test_unknown_row_rate': unknown_rows / report['datasets']['test']['rows']}
        finally:
            db.close()
    _write_json(out, report)
    return report


def read_package_ids(path: str | Path) -> pd.Series:
    path = Path(path)
    if path.suffix == '.parquet':
        values = pd.read_parquet(path, columns=['pkg_no'])['pkg_no'].astype('string')
    else:
        values = pd.read_csv(path, dtype={'pkg_no': 'string'}, keep_default_na=False)['pkg_no']
    if values.empty or values.isna().any() or values.eq('').any() or values.duplicated().any():
        raise ValueError('Holdout key list must be nonempty, nonmissing and unique')
    return values


def audit_development(data_dir: str | Path, exclude_ids: str | Path, out: str | Path,
                      *, batch_size=200_000, label_tolerance_hours=1 / 3600) -> dict:
    data_dir, exclude_ids, out = Path(data_dir).resolve(), Path(exclude_ids).resolve(), Path(out).resolve()
    source_manifest = _core_manifest(data_dir)
    holdout = set(read_package_ids(exclude_ids).tolist())
    included, excluded, matched = [], 0, set()
    columns = ['pkg_no', 'pay_time', 'issue_time', 'sign_time', 'y_total_hours',
               'lgs_company_nm', 'shpp_chnl', 'shpp_prvn_nm']
    for batch in pq.ParquetFile(data_dir / 'train.parquet').iter_batches(batch_size=batch_size, columns=columns):
        frame = batch.to_pandas()
        mask = frame['pkg_no'].isin(holdout)
        excluded += int(mask.sum())
        matched.update(frame.loc[mask, 'pkg_no'].tolist())
        # Never calculate a statistic, residual or example on held-out targets.
        included.append(frame.loc[~mask].drop(columns='pkg_no'))
    if matched != holdout:
        raise ValueError('Holdout IDs must all belong to the prepared training dataset')
    frame = pd.concat(included, ignore_index=True)
    if frame.empty:
        raise ValueError('No development rows remain')
    label = frame['y_total_hours']
    actual = (frame['sign_time'] - frame['pay_time']).dt.total_seconds() / 3600
    valid = label.notna() & actual.notna()
    mismatch = (label[valid] - actual[valid]).abs() > label_tolerance_hours
    def summarize(group):
        y = group['y_total_hours'].dropna()
        return {'n': int(len(group)), 'label_nonnull': int(len(y)),
                'mean': float(y.mean()) if len(y) else None,
                **{f'Q{int(q*100)}': float(y.quantile(q)) if len(y) else None for q in [.5,.9,.97,.99]},
                'over_30_days_rate': float((y > 720).mean()) if len(y) else None}
    groups = {}
    for column in ['lgs_company_nm', 'shpp_chnl', 'shpp_prvn_nm', 'pay_time', 'issue_time']:
        keys = frame[column].dt.strftime('%Y-%m-%d') if column in TIME_COLUMNS else frame[column]
        groups[column] = {str(key): summarize(group) for key, group in frame.groupby(keys, dropna=False)}
    report = {'created_at': _timestamp(), 'mode': 'development',
        'holdout_key_sha256': _sha256(exclude_ids), 'excluded_holdout_rows': excluded,
        'development_rows': len(frame), 'summary': summarize(frame), 'groups': groups,
        'label_time_consistency': {'tolerance_hours': label_tolerance_hours,
            'comparable_rows': int(valid.sum()), 'mismatches': int(mismatch.sum()),
            'missing_label': int(label.isna().sum()), 'missing_sign_time': int(frame['sign_time'].isna().sum()),
            'label_outside_0_to_2160_hours': int(((label <= 0) | (label >= 2160)).sum())},
        'preparation_manifest_sha256': _sha256(data_dir / 'preparation_manifest.json'),
        'training_source_sha256': source_manifest['files']['train']['source_sha256'],
        'holdout_label_statistics_emitted': False}
    _write_json(out, report)
    return report
