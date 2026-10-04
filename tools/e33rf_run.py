"""三seed复现先行；RF2独立于RF1，RF3逐seed完整Pilot Gate决定。"""
from models.apacc_features import load_dino
from tools.e33rf_common import *
from tools.e33rf_prepare import prepare,cache_rotations
from tools.e33rf_evaluate import evaluate
from tools.e33rf_gpu_audit import audit
from tools.e33rf_train import train

def run():
    prepare();dino,_=load_dino('cuda',WEIGHTS);cache_rotations(dino,False);audit(dino)
    reproduced=[]
    for seed in SEEDS:
        dest=OUT/'RF0_reproduction'/('seed%d'%seed)
        if not (dest/'reproduction_gate.json').exists():
            model,parameters=build(seed);record=evaluate(model,seed,dest,dino)
            baseline=read(RC/'RC0_zero_shot'/('seed%d'%seed)/'real/summary.json');real=read(dest/'real/summary.json')
            checks=dict(rot90=abs(real['rot90_success']['mean']-baseline['rot90_success']['mean'])<=.02,
                near=abs(real['near_advantage']['mean']-baseline['near_advantage']['mean'])<=.5,
                match=abs(real['errors']['matched']['mean']-baseline['errors']['matched']['mean'])<=.5,
                controlled=record['controlled_clean']>=.99)
            write(dest/'reproduction_gate.json',dict(checks=checks,**{'pass':all(checks.values())},parameters=parameters))
            del model;torch.cuda.empty_cache()
        reproduced.append(read(dest/'reproduction_gate.json')['pass'])
    write(OUT/'rf0_status.json',dict(seeds=SEEDS,checks=reproduced,**{'pass':all(reproduced)}))
    if all(reproduced):
        cache_rotations(dino,True)
        for seed in SEEDS:
            train(seed,'RF1',dino);rf2=train(seed,'RF2',dino)
            if rf2['pilot']['pass']:train(seed,'RF3',dino)
            else:write(folder(seed,'RF2')/'RF3_not_run.json',dict(reason='complete Pilot Gate failed',checks=rf2['pilot']['checks']))
        for variant in ABLATIONS:train(42,'RF2',dino,variant)
    del dino;torch.cuda.empty_cache()
    from tools.e33rf_report import report
    report()
if __name__=='__main__':run()
