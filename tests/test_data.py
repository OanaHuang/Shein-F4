import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import pandas as pd
import pyarrow.parquet as pq
from f4.data import (SHARED_COLUMNS, TRAIN_COLUMNS, prepare_data,
                     audit_structure, audit_development)


from f4.platform_utils import file_digest


class DataPipelineSmokeTest(unittest.TestCase):
    def make_raw(self, root, duplicate=False, missing_key=False):
        raw = root / 'raw'; raw.mkdir()
        rows = []
        for i in range(4):
            row = {column: '' for column in TRAIN_COLUMNS}
            row.update(pkg_no=f'{i:04d}', pkg_goods_cnt='2', weight='100.5',
                       pay_time='2026-08-01 00:00:00', issue_time='2026-08-01 01:00:00',
                       sign_time=f'2026-08-0{i+2} 00:00:00', y_total_hours=str((i+1)*24),
                       lgs_company_nm='L1', shpp_chnl='C1', shpp_prvn_nm='P1')
            rows.append(row)
        if duplicate: rows[3]['pkg_no'] = rows[0]['pkg_no']
        if missing_key: rows[3]['pkg_no'] = ''
        rows[1]['weight'] = 'not-a-number'
        pd.DataFrame(rows,columns=TRAIN_COLUMNS).to_csv(raw/'train_202608_0920.csv',index=False)
        test = [{c: rows[0][c] for c in SHARED_COLUMNS}]
        test[0].update(pkg_no='0999',lgs_company_nm='NEW')
        pd.DataFrame(test,columns=SHARED_COLUMNS).to_csv(raw/'test_202609.csv',index=False)
        return raw

    def prepare(self, root, **kw):
        return prepare_data(self.make_raw(root, **kw),root/'core',chunk_size=2,
                            expected_rows={'train':4,'test':1})

    def test_core_preserves_keys_and_reports_conversion_without_deleting_rows(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); manifest=self.prepare(root)
            frame=pd.read_parquet(root/'core/train.parquet')
            self.assertEqual(frame['pkg_no'].tolist(),['0000','0001','0002','0003'])
            self.assertEqual(len(frame.columns),25)
            self.assertNotIn('pack_time',frame)
            self.assertTrue(pd.isna(frame.loc[1,'weight']))
            self.assertEqual(manifest['files']['train']['normalization_anomalies']['weight_invalid_numeric'],1)
            self.assertEqual(manifest['rows_removed'],0)

    def test_duplicate_across_chunks_and_missing_keys_fail(self):
        for kind in ['duplicate','missing_key']:
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as d:
                with self.assertRaises(ValueError): self.prepare(Path(d),**{kind:True})

    def test_cross_dataset_duplicate_keys_fail(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); raw=self.make_raw(root)
            path=raw/'test_202609.csv'; test=pd.read_csv(path,dtype='string')
            test['pkg_no']='0000'; test.to_csv(path,index=False)
            with self.assertRaises(ValueError):
                prepare_data(raw,root/'core',chunk_size=2,expected_rows={'train':4,'test':1})

    def test_label_legal_range_is_open_and_rows_are_retained(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); raw=self.make_raw(root)
            path=raw/'train_202608_0920.csv'
            frame=pd.read_csv(path,dtype='string',keep_default_na=False)
            frame.loc[0,'y_total_hours']='0'
            frame.loc[1,'y_total_hours']='2160'
            frame.to_csv(path,index=False)
            manifest=prepare_data(raw,root/'core',chunk_size=2,
                                  expected_rows={'train':4,'test':1})
            self.assertEqual(manifest['files']['train']['normalization_anomalies']
                             ['label_outside_0_to_2160_hours'],2)
            self.assertEqual(manifest['files']['train']['rows'],4)

    def test_structure_omits_target_and_reports_unknown_category(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.prepare(root)
            report=audit_structure(root/'core',root/'structure.json',batch_size=2,
                                   distinct_memory_limit=1)
            self.assertFalse(report['target_labels_inspected'])
            self.assertNotIn('y_total_hours',report['datasets']['train']['fields'])
            self.assertNotIn('sign_time',report['datasets']['train']['fields'])
            self.assertEqual(report['category_coverage']['lgs_company_nm']['test_unknown_rows'],1)
            self.assertEqual(report['datasets']['train']['daily_rows']['issue_time'],{'2026-08-01':4})

    def test_development_audit_is_invariant_to_holdout_label_changes(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.prepare(root)
            holdout=root/'holdout.csv';pd.DataFrame({'pkg_no':['0003']}).to_csv(holdout,index=False)
            first=audit_development(root/'core',holdout,root/'dev1.json',batch_size=2)
            path=root/'core/train.parquet';table=pq.read_table(path).to_pandas()
            table.loc[table.pkg_no=='0003','y_total_hours']=9999
            table.loc[table.pkg_no=='0003','sign_time']=pd.Timestamp('2029-01-01')
            table.to_parquet(path,index=False)
            # The altered copy gets a new cache checksum; raw sources remain intact.
            import hashlib
            manifest_path=root/'core/preparation_manifest.json'
            manifest=json.loads(manifest_path.read_text())
            with path.open('rb') as stream:
                manifest['files']['train']['parquet_sha256']=file_digest(stream, 'sha256').hexdigest()
            manifest_path.write_text(json.dumps(manifest))
            second=audit_development(root/'core',holdout,root/'dev2.json',batch_size=2)
            self.assertEqual(first['summary'],second['summary'])
            self.assertEqual(first['groups'],second['groups'])
            self.assertEqual(first['label_time_consistency'],second['label_time_consistency'])


if __name__=='__main__': unittest.main()
