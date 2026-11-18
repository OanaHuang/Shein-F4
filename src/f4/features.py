"""Issue-time features with an explicit, frozen shared-field allowlist.

Select ``SHARED_COLUMNS`` from a core table before calling this module. Passing
labels or fulfilment timestamps is an error, even if they would be ignored.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence

import numpy as np
import pandas as pd

SHARED_COLUMNS = (
    'pkg_no', 'pkg_goods_cnt', 'weight', 'shpp_country_nm', 'shpp_prvn_nm',
    'shpp_city_nm', 'stk_cate_tp_nm', 'shpp_chnl', 'lgs_company_nm',
    'combine_flag', 'combine_type', 'start_airport_cd', 'end_airport_cd',
    'mode_category', 'pay_mthd', 'business_tp', 'operation_type', 'route_cat',
    'sale_area', 'send_country_nm', 'shpp_post_no', 'pay_time', 'issue_time',
)
CATEGORY_COLUMNS = tuple(c for c in SHARED_COLUMNS if c not in {
    'pkg_no', 'shpp_post_no', 'weight', 'pkg_goods_cnt', 'pay_time', 'issue_time',
})
NUMERIC_FEATURES = (
    'weight', 'pkg_goods_cnt', 'weight_log1p', 'weight_per_item',
    'weight_missing', 'weight_invalid', 'goods_count_missing',
    'goods_count_invalid', 'pay_hour', 'pay_weekday', 'issue_hour',
    'issue_weekday', 'elapsed_hours', 'elapsed_negative', 'pay_time_missing',
    'issue_time_missing',
)
CATEGORICAL_FEATURES = tuple('cat__' + c for c in CATEGORY_COLUMNS)
FEATURE_COLUMNS = NUMERIC_FEATURES + tuple(
    feature for column in CATEGORY_COLUMNS
    for feature in ('cat__' + column, 'unknown__' + column, 'missing__' + column)
)


def validate_pkg_ids(values: Sequence, expected_rows: int | None = None,
                     name: str = 'pkg_ids') -> list[str]:
    """Reject ambiguous or duplicated keys instead of coercing away leading zeroes."""
    result = list(values)
    if expected_rows is not None and len(result) != expected_rows:
        raise ValueError(f'{name} length does not match rows')
    if not result:
        raise ValueError(f'{name} must not be empty')
    if any(not isinstance(value, str) or value == '' for value in result):
        raise ValueError(f'{name} must contain nonempty strings')
    if len(set(result)) != len(result):
        raise ValueError(f'{name} contains duplicate keys')
    return result


def pkg_ids_sha256(values: Sequence) -> str:
    """Order-sensitive digest; lengths make concatenated keys unambiguous."""
    digest = hashlib.sha256()
    for value in values:
        encoded = value.encode('utf-8')
        digest.update(len(encoded).to_bytes(8, 'big'))
        digest.update(encoded)
    return digest.hexdigest()


def _category(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    values = series.astype('string')
    missing = values.isna() | values.eq('').fillna(False)
    return values, missing


class FeatureBuilder:
    """Fit category dictionaries only on the supplied fitting cohort.

    Category codes are 0=missing, 1=unknown, and >=2=known fitting values.
    Unknown/missing flags remain separate numeric features. ``pkg_no`` and
    postal codes never enter the returned model matrix.
    """

    schema_version = 'shared_v1'
    categorical_features = CATEGORICAL_FEATURES

    def _encoder_digest(self) -> str:
        state = dict(schema_version=self.schema_version, category_maps=self.category_maps_,
                     fit_ids_sha256=self.fit_ids_sha256_)
        return hashlib.sha256(json.dumps(state, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

    @staticmethod
    def _validate(raw: pd.DataFrame) -> list[str]:
        if not isinstance(raw, pd.DataFrame) or raw.columns.duplicated().any():
            raise ValueError('Feature input must be a DataFrame with unique columns')
        extra = sorted(set(raw.columns) - set(SHARED_COLUMNS))
        missing = sorted(set(SHARED_COLUMNS) - set(raw.columns))
        if extra:
            raise ValueError(f'Forbidden/non-shared feature columns: {extra}')
        if missing:
            raise ValueError(f'Missing shared columns: {missing}')
        return validate_pkg_ids(raw['pkg_no'].tolist(), len(raw), 'pkg_no')

    def fit(self, raw: pd.DataFrame) -> 'FeatureBuilder':
        ids = self._validate(raw)
        self.category_maps_ = {}
        for column in CATEGORY_COLUMNS:
            values, missing = _category(raw[column])
            unique = sorted(values.loc[~missing].unique().tolist())
            self.category_maps_[column] = {value: index + 2 for index, value in enumerate(unique)}
        self.fit_ids_sha256_ = pkg_ids_sha256(ids)
        self.fit_rows_ = len(raw)
        self.encoder_sha256_ = self._encoder_digest()
        self.feature_names_ = list(FEATURE_COLUMNS)
        return self

    def transform(self, raw: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, 'category_maps_'):
            raise RuntimeError('FeatureBuilder has not been fitted')
        ids = self._validate(raw)
        values = {}
        weight = pd.to_numeric(raw['weight'], errors='coerce').astype('float64')
        count = pd.to_numeric(raw['pkg_goods_cnt'], errors='coerce').astype('float64')
        good_weight = np.isfinite(weight) & (weight > 0)
        good_count = np.isfinite(count) & (count > 0) & (count == np.floor(count))
        clean_weight = weight.where(good_weight)
        clean_count = count.where(good_count)
        values.update(
            weight=clean_weight,
            pkg_goods_cnt=clean_count,
            weight_log1p=np.log1p(clean_weight),
            weight_per_item=clean_weight / clean_count,
            weight_missing=raw['weight'].isna().astype('uint8'),
            weight_invalid=(~good_weight & raw['weight'].notna()).astype('uint8'),
            goods_count_missing=raw['pkg_goods_cnt'].isna().astype('uint8'),
            goods_count_invalid=(~good_count & raw['pkg_goods_cnt'].notna()).astype('uint8'),
        )
        pay = pd.to_datetime(raw['pay_time'], errors='coerce')
        issue = pd.to_datetime(raw['issue_time'], errors='coerce')
        if pay.dt.tz is not None or issue.dt.tz is not None:
            raise ValueError('Timezone-aware inputs require an explicit project time protocol')
        elapsed = (issue - pay).dt.total_seconds() / 3600
        values.update(
            pay_hour=pay.dt.hour, pay_weekday=pay.dt.day_of_week,
            issue_hour=issue.dt.hour, issue_weekday=issue.dt.day_of_week,
            elapsed_hours=elapsed.where(elapsed >= 0),
            elapsed_negative=(elapsed < 0).astype('uint8'),
            pay_time_missing=pay.isna().astype('uint8'),
            issue_time_missing=issue.isna().astype('uint8'),
        )
        diagnostics = {}
        for column in CATEGORY_COLUMNS:
            category, missing = _category(raw[column])
            mapped = category.map(self.category_maps_[column])
            unknown = ~missing & mapped.isna()
            values['cat__' + column] = mapped.where(~missing, 0).fillna(1).astype('int32')
            values['unknown__' + column] = unknown.astype('uint8')
            values['missing__' + column] = missing.astype('uint8')
            diagnostics[column] = {'unknown': int(unknown.sum()), 'missing': int(missing.sum())}
        result = pd.DataFrame(values, index=raw.index).loc[:, list(FEATURE_COLUMNS)]
        for column in NUMERIC_FEATURES:
            # Extremely large finite inputs may overflow float32. Treat them as
            # missing rather than passing infinities to the learner.
            with np.errstate(over='ignore', invalid='ignore'):
                result[column] = result[column].astype('float32')
            result[column] = result[column].where(np.isfinite(result[column]))
        result.attrs.update(
            feature_schema_version=self.schema_version,
            feature_fit_ids_sha256=self.fit_ids_sha256_,
            feature_encoder_sha256=self.encoder_sha256_,
            pkg_ids_sha256=pkg_ids_sha256(ids),
            unknown_categories=diagnostics,
        )
        self.last_transform_summary_ = diagnostics
        return result

    def fit_transform(self, raw: pd.DataFrame) -> pd.DataFrame:
        return self.fit(raw).transform(raw)

    def to_dict(self) -> dict:
        if not hasattr(self, 'category_maps_'):
            raise RuntimeError('FeatureBuilder has not been fitted')
        return dict(schema_version=self.schema_version, category_maps=self.category_maps_,
                    fit_ids_sha256=self.fit_ids_sha256_, fit_rows=self.fit_rows_,
                    encoder_sha256=self.encoder_sha256_)

    @classmethod
    def from_dict(cls, state: dict) -> 'FeatureBuilder':
        if state.get('schema_version') != cls.schema_version:
            raise ValueError('Unsupported feature schema version')
        if set(state['category_maps']) != set(CATEGORY_COLUMNS):
            raise ValueError('Category dictionary schema mismatch')
        result = cls()
        result.category_maps_ = state['category_maps']
        result.fit_ids_sha256_ = state['fit_ids_sha256']
        result.fit_rows_ = int(state['fit_rows'])
        result.feature_names_ = list(FEATURE_COLUMNS)
        result.encoder_sha256_ = result._encoder_digest()
        if result.encoder_sha256_ != state.get('encoder_sha256'):
            raise ValueError('Encoder dictionary digest mismatch')
        return result
