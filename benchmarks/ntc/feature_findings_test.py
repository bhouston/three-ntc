"""Statistical reporting checks; requires matplotlib in the test interpreter."""
import json, math, subprocess, sys, tempfile, unittest
from pathlib import Path

class FeatureFindingsTests(unittest.TestCase):
    def run_fixture(self, mismatch=False):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);(out/'results').mkdir()
            cases=[]
            for seed in [1,2,3]:
                for variant in ['reference','activation-relu','hidden-16']:
                    cases.append(dict(material='brick',variant={'id':variant},seed=seed))
                    if variant=='hidden-16':continue # missing contrasts remain in the Holm family
                    delta=seed if variant=='activation-relu' else 0
                    mse=.001*10**(-delta/10)
                    q=dict(psnr=30+delta,mip0Psnr=30+delta,equalMipPsnr=30+delta,normalizedMse=mse,
                        mips=[dict(pixels=1,perComponent=[dict(normalizedMse=mse)])])
                    r=dict(status='complete',sourceHash='fixture',completedIterations=4000,samples=16384000,
                        referenceHashes=['source'],adapter={'vendor':'test'},gradientPrecision='float',
                        material='brick',variant={'id':variant},seed=seed,quality={'exported':q},
                        channels=[dict(key='albedo',offset=0,size=1)],trainingMs=1,
                        storage=dict(jsonBytes=100,brotliBytes=50,runtimeGridBytes=100))
                    if mismatch and seed==3 and variant=='activation-relu':r['referenceHashes']=['different']
                    (out/'results'/f'{variant}-{seed}.json').write_text(json.dumps(r))
            plan=dict(sourceHash='fixture',seeds=[1,2,3],iterations=4000,resolution=1024,cases=cases)
            (out/'plan.json').write_text(json.dumps(plan))
            subprocess.run([sys.executable,str(Path(__file__).with_name('feature-findings.py')),str(out)],check=True,capture_output=True)
            return json.loads((out/'feature-effects.json').read_text())['effects'][0]

    def test_paired_direction_interval_and_planned_family_correction(self):
        e=self.run_fixture()
        self.assertEqual(e['values'],[1,2,3]);self.assertEqual(e['mean'],2)
        self.assertAlmostEqual(e['low'],2-4.302652730/math.sqrt(3))
        self.assertAlmostEqual(e['geometricMseReductionPercent'],100*(1-10**(-.2)))
        self.assertAlmostEqual(e['holmP'],2*(1-math.sqrt(12/14)))
        self.assertTrue(e['complete'])

    def test_mismatched_source_is_excluded_from_pairing(self):
        e=self.run_fixture(mismatch=True)
        self.assertEqual(e['n'],2);self.assertEqual(e['mean'],1.5)
        self.assertFalse(e['complete']);self.assertNotIn('holmP',e)

if __name__=='__main__':unittest.main()
