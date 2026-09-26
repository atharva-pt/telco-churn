"""Exercise the installed package from outside the repository, in fresh processes."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from tests.fixtures import make_frame
from telco_churn.artifacts import load_bundle
from telco_churn.config import TrainingConfig
from telco_churn.data import FEATURE_COLUMNS, read_dataset


class WorkflowTests(unittest.TestCase):
    def test_fresh_process_workflow_and_failure_exit_codes(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            data=root/'data.csv'; config=root/'config.json'
            make_frame().to_csv(data,index=False)
            config.write_text(json.dumps(TrainingConfig(cv_folds=2,logistic_c=(1,),forest_trees=5,
                                                        forest_depths=(3,),forest_min_leaf=(1,)).to_dict()))
            def call(*args,code=0):
                result=subprocess.run([sys.executable,'-m','telco_churn',*map(str,args)],cwd=root,
                                      capture_output=True,text=True,timeout=60)
                self.assertEqual(result.returncode,code,result.stdout+result.stderr)
                return result
            explore=root/'exploration'; run=root/'run'
            call('validate','--data',data)
            call('explore','--data',data,'--config',config,'--output',explore)
            call('train','--data',data,'--config',config,'--split-manifest',explore/'split.json','--run-dir',run)
            self.assertTrue((run/'COMPLETE').is_file())
            self.assertFalse((run/'evaluation').exists())
            call('evaluate','--data',data,'--run-dir',run)
            self.assertTrue((run/'evaluation'/'COMPLETE').is_file())
            call('evaluate','--data',data,'--run-dir',run,code=2)
            batch=root/'batch.csv'; output=root/'scores.csv'
            make_frame(3).drop(columns='Churn').to_csv(batch,index=False)
            call('predict','--data',batch,'--run-dir',run,'--output',output)
            scores=pd.read_csv(output,dtype={'customerID':str})
            self.assertEqual(scores.customerID.tolist(),['00000','00001','00002'])
            model,_=load_bundle(run)
            expected=model.predict_proba(read_dataset(batch,labeled=False)[FEATURE_COLUMNS])[:,1]
            np.testing.assert_allclose(scores.churn_probability,expected,rtol=1e-10,atol=1e-12)
            call('predict','--data',batch,'--run-dir',run,'--output',output,code=2)
            batch.write_text('customerID\nsecret-customer-record\n')
            failure=call('validate','--data',batch,code=2)
            self.assertNotIn('secret-customer-record',failure.stderr)
            call('explore','--data',data,'--config',config,'--output',explore,code=2)
            call('train','--data',data,'--config',config,'--split-manifest',explore/'split.json','--run-dir',run,code=2)
