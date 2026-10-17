"""后备必须等真实门禁/全部800完成；不重启已清理的轻量作业。"""
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch
from tools import m54_fallback_dispatch as dispatch


class FallbackDispatchTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);(self.root/'resources').mkdir()
        self.calls=[];self.jobs=[];self.states={}
        self.runtime=types.SimpleNamespace(require_protocol=lambda:None)
        self.patches=[patch.object(dispatch,'OUT',self.root),patch.dict('sys.modules',{'tools.m54_fallback_run':self.runtime,
            'fcntl':types.SimpleNamespace(flock=lambda *a:None,LOCK_EX=1)}),
            patch.object(dispatch.subprocess,'check_output',side_effect=self.accounting),
            patch.object(dispatch,'submit',side_effect=self.submit),patch.object(dispatch,'share_results')]
        for p in self.patches:p.start()
        self.addCleanup(self.temp.cleanup)
        for p in self.patches:self.addCleanup(p.stop)

    def save(self,path,value):
        path=self.root/path;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value),encoding='utf-8')

    def accounting(self,*args,**kwargs):return ''.join(str(j['id'])+'|'+self.states[j['id']]+'|\n' for j in self.jobs)
    def submit(self,name,action,gpu,**kwargs):
        self.calls.append((name,action,gpu,kwargs));return 100+len(self.calls)
    def add(self,name,state='COMPLETED'):
        identity=len(self.jobs)+1;self.jobs.append(dict(name=name,id=identity));self.states[identity]=state
        self.save('resources/jobs.json',self.jobs)
    def passed(self):
        self.add('FB_ROUND0');self.save('fallback/audit/round0_complete.json',dict(passed=True))

    def test_only_real_gpu_round0_is_submitted_first(self):
        self.add('DEV_EVAL')
        dispatch.advance()
        self.assertEqual([(c[0],c[1]) for c in self.calls],[('FB_ROUND0','fallback_audit')])

    def test_pending_round0_never_starts_training(self):
        self.add('FB_ROUND0','PENDING');self.save('fallback/audit/round0_complete.json',dict(passed=True))
        dispatch.advance();self.assertEqual(self.calls,[])

    def test_both_routes_get_exact_800_after_round0(self):
        self.passed();dispatch.advance()
        self.assertEqual([c[0] for c in self.calls],['FB_TRAIN_M_L1','FB_TRAIN_M_L2'])
        self.assertTrue(all(c[1]=='fallback_train' and c[3]['extra'][-2:]==['--updates','800'] for c in self.calls))

    def test_one_training_pending_blocks_unified_pilot(self):
        self.passed();self.add('FB_TRAIN_M_L1');self.add('FB_TRAIN_M_L2','PENDING')
        for arm in dispatch.ARMS:self.save('training/'+arm+'/complete0800.json',dict(updates=800))
        dispatch.advance();self.assertEqual(self.calls,[])

    def test_finished_training_submits_two_pilots_without_early_evaluation(self):
        self.passed()
        for arm in dispatch.ARMS:
            self.add('FB_TRAIN_'+arm);self.save('training/'+arm+'/complete0800.json',dict(updates=800))
            self.save('fallback/audit/'+arm+'_training800.json',dict(passed=True))
        dispatch.advance()
        self.assertEqual([c[0] for c in self.calls],['FB_PILOT_M_L1','FB_PILOT_M_L2'])
        self.assertTrue(all(c[1]=='fallback_generate' for c in self.calls))

    def test_latest_engineering_failure_blocks_new_scientific_jobs(self):
        self.passed();self.add('FB_TRAIN_M_L1','FAILED');dispatch.advance();self.assertEqual(self.calls,[])


if __name__=='__main__':unittest.main()
