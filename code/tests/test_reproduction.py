import argparse
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from reproduce import aggregate_records, load_cfg, metric_values, selected


class FactorGradReproductionTests(unittest.TestCase):
    def test_cfg_is_literal_only(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'cfg'
            p.write_text("Namespace(images='images',eval=True,resolution=-1)")
            self.assertEqual(load_cfg(p)['resolution'],-1)
            p.write_text("Namespace(eval=__import__('os').getcwd())")
            with self.assertRaises(ValueError):
                load_cfg(p)

    def test_missing_or_empty_metrics_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)
            with self.assertRaises(FileNotFoundError):
                metric_values(p)
            (p/'results.json').write_text('{}')
            with self.assertRaises(RuntimeError):
                metric_values(p)

    def test_time_is_scene_sum_then_execution_mean(self):
        records=[]
        for rep,times in [('R1',[10,20]),('R2',[30,40])]:
            for scene,time in zip(['a','b'],times):
                records.append(dict(group='primary',method='full',dataset='mipnerf360',repetition=rep,scene=scene,psnr=30.,ssim=.9,lpips=.1,wall_time_s=time))
        with tempfile.TemporaryDirectory() as directory:
            rows=aggregate_records(records,Path(directory))
            row=next(r for r in rows if r['dataset']=='all13')
            self.assertEqual(row['wall_time_sum_s_mean'],50)
            self.assertEqual(row['psnr_mean'],30.)

    def test_manifest_scope(self):
        root=Path(__file__).resolve().parents[1]
        runs=json.loads((root/'configs/runs.json').read_text())
        args=argparse.Namespace(suite='table1',run_id=None,scenes=None)
        self.assertEqual(sum(selected(r,args) for r in runs),182)
        args.suite='selected'
        self.assertEqual(sum(selected(r,args) for r in runs),13)
        self.assertEqual(sum(len(r['bundled_weights']) for r in runs),0)
        for r in runs:
            self.assertTrue(r['weights'])
            self.assertTrue(r['train_args'])
            self.assertFalse(any('/home/' in a for a in r['train_args']))


if __name__=='__main__':
    unittest.main()
