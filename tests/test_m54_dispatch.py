"""调度闭环的虚拟文件/Slurm检查；不执行任何模型、不产生科研更新。"""
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch
from tools import m54_dispatch as dispatcher,m54_protocol as protocol,m54_selection as selection


class DispatchTests(unittest.TestCase):
    def test_full_allowed_path_limits_parallelism_and_uses_gpu_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'resources').mkdir();jobs=[dict(name='ROUND0',id=1),dict(name='ROUND0_R1',id=2,logical_name='ROUND0')]
            states={1:'FAILED',2:'COMPLETED'};submitted=[]
            protocol.write(root/'resources/jobs.json',jobs)
            protocol.write(root/'audit/round0_complete.json',dict(passed=True))
            for stage in ['pilot','lock','dev32','natural','oracle_diagnostic']:
                path=root/'splits'/(stage+'.json');protocol.write(path,[dict(id='unit_fixture')])
                if stage in ['pilot','lock']:
                    protocol.write(root/'audit/input_review'/(stage+'_approval.json'),dict(passed=True,reviewer='assistant',
                        actual_reviewed_ids=['unit_fixture'],split_sha256=protocol.sha(path)))
            def submit(name,action,gpu,dependencies=(),extra=(),**unused):
                self.assertLess(sum(s=='PENDING' for s in states.values()),4)
                identity=len(jobs)+1;job=dict(name=name,id=identity,action=action,gpu=gpu,extra=list(extra))
                jobs.append(job);states[identity]='PENDING';submitted.append(job);protocol.write(root/'resources/jobs.json',jobs)
                return identity
            def share(source,target,arms):
                for arm in arms:
                    protocol.write(root/target/(arm+'_complete.json'),protocol.read(root/source/(arm+'_complete.json')))
            def top4():
                value=dict(ready=True,selected=['M1','M9']);protocol.freeze(root/'selection/top4.json',value);return value
            def combinations():
                value=dict(recipes={'C_M1_M9':dict(components=[dict(arm='M1',updates=2400),dict(arm='M9',updates=2400)])})
                protocol.freeze(root/'selection/combinations.json',value);return value
            def passers(stage,filename,limit=None):
                value=dict(selected=['M1'] if filename=='dev_passers.json' else ['C_M1_M9'])
                protocol.freeze(root/'selection'/filename,value);return value
            def finish_pending():
                for job in jobs:
                    if states[job['id']]!='PENDING':continue
                    args=job['extra'];arg=lambda key,default=None:args[args.index(key)+1] if key in args else default
                    action=job['action']
                    if action=='generate':
                        stage=protocol.result_stage(arg('--stage'),int(arg('--updates','800')),int(arg('--seed','42')))
                        protocol.write(root/stage/(arg('--arm')+'_complete.json'),dict(n=1,arm=arg('--arm')))
                    elif action=='train':
                        protocol.write(root/'training'/arg('--arm')/('complete%04d.json'%int(arg('--updates','800'))),dict(updates=int(arg('--updates','800'))))
                    elif action=='evaluate':
                        stage=arg('--stage');protocol.write(root/stage/'evaluation_complete.json',dict(complete=True))
                        if stage=='round1':
                            protocol.write(root/stage/'gates.json',{a:dict(passed=True) for a in ['M8_T100','M8_T085','M8_T070']})
                            protocol.write(root/stage/'means.json',{a:dict(lpips_gt=.1) for a in ['M8_T100','M8_T085','M8_T070']})
                    elif action=='finalize':protocol.write(root/'audit/final_complete.json',dict(passed=True))
                    states[job['id']]='COMPLETED'
            fcntl=types.SimpleNamespace(LOCK_EX=1,flock=lambda *args:None)
            with patch.object(dispatcher,'OUT',root),patch.object(protocol,'OUT',root),patch.object(selection,'OUT',root),\
                 patch.object(dispatcher,'submit',submit),patch.object(dispatcher.subprocess,'check_output',lambda *a,**k:'\n'.join(str(i)+'|'+state for i,state in states.items())),\
                 patch.object(selection,'share_results',share),patch.object(selection,'freeze_top4',top4),\
                 patch.object(selection,'freeze_combinations',combinations),patch.object(selection,'freeze_passers',passers),\
                 patch.dict(sys.modules,{'fcntl':fcntl}):
                for _ in range(40):
                    dispatcher.advance();finish_pending()
                    if any(j['name']=='FINAL' for j in submitted):
                        dispatcher.advance();break
                else:self.fail('调度未抵达最终核验，可能有阶段名称或依赖死锁')
            self.assertEqual(len([j for j in submitted if j['name'].startswith('TRAIN_')]),8)
            self.assertEqual({j['name'] for j in submitted if j['name'].startswith('DEEPEN_')},{'DEEPEN_M1','DEEPEN_M9'})
            self.assertTrue(all(j['gpu'] for j in submitted if j['action']=='evaluate'))
            self.assertTrue(any(j['name'].startswith('SEED123_') for j in submitted))
            self.assertTrue(any(j['name'].startswith('SEED456_') for j in submitted))
            self.assertEqual(protocol.read(root/'decision.json')['local_delivery'],'PENDING_DOWNLOAD_AND_VERIFICATION')
            self.assertFalse(protocol.read(root/'decision.json')['complete'])


if __name__=='__main__':unittest.main()
