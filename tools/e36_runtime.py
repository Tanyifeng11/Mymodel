"""仅绑定当前实例，在真实 pre-concat 点修改首次消费的 skip。"""
import ast
import collections
import inspect
import textwrap
from types import MethodType


class SkipBridge:
    def __init__(self,unet):
        self.unet=unet;self.adapter=None;self.sketch=None;self.trace=[];self.calls=collections.Counter()
        self.branch='unknown';self.index=0;self.block=unet.up_blocks[2];self.original=self.block.forward
        source=inspect.getsource(type(self.block).forward);self.source=source
        tree=ast.parse(textwrap.dedent(source));count=[0]
        class Insert(ast.NodeTransformer):
            def visit_Assign(this,node):
                this.generic_visit(node);v=node.value
                if (isinstance(v,ast.Call) and isinstance(v.func,ast.Attribute) and v.func.attr=='cat'
                    and v.args and isinstance(v.args[0],(ast.List,ast.Tuple))
                    and [getattr(n,'id',None) for n in v.args[0].elts]==['hidden_states','res_hidden_states']):
                    count[0]+=1
                    return [ast.parse('res_hidden_states = _e36_apply(hidden_states, res_hidden_states, temb)').body[0],node]
                return node
        tree=Insert().visit(tree);ast.fix_missing_locations(tree);assert count[0]==1
        namespace=dict(type(self.block).forward.__globals__);namespace['_e36_apply']=self.apply
        exec(compile(tree,'<E36 pre-concat>','exec'),namespace)
        self.block.forward=MethodType(namespace['forward'],self.block)
        def before(module,args,kwargs):
            self.index=0
            timestep=args[1] if len(args)>1 else kwargs['timestep']
            self.timestep=int(timestep.flatten()[0]) if hasattr(timestep,'flatten') else int(timestep)
            self.branch='conditional' if 'sa_hidden_states' in (kwargs.get('cross_attention_kwargs') or {}) else 'unconditional'
        self.handle=unet.register_forward_pre_hook(before,with_kwargs=True)

    def apply(self,hidden,skip,temb):
        index=self.index;self.index+=1;self.calls['%s/%d'%(self.branch,index)]+=1
        if len(self.trace)<12:
            self.trace.append(dict(branch=self.branch,index=index,timestep=self.timestep,
                shape=list(skip.shape),hidden_shape=list(hidden.shape),concat_shape=[skip.shape[0],hidden.shape[1]+skip.shape[1],*skip.shape[2:]],
                dtype=str(skip.dtype),skip_rms=float(skip.float().square().mean().sqrt()),time_shape=list(temb.shape),
                sketch_input='original normalized sketch' if self.branch=='conditional' else 'not consumed; adapter OFF',
                                   path='up_blocks.2/pre_concat',applied=self.adapter is not None and index==0 and self.branch=='conditional'))
        if index!=0 or self.adapter is None or self.branch!='conditional':return skip
        assert tuple(skip.shape[-2:])==(32,24)
        return self.adapter(skip,self.sketch,temb)

    def reset(self):self.calls.clear();self.trace=[]
    def report(self):return dict(counts=dict(self.calls),trace=self.trace)
    def close(self):self.block.forward=self.original;self.handle.remove()
