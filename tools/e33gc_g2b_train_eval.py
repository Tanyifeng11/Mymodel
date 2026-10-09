"""同一GPU分配顺序完成固定训练和只读评估；子进程退出后释放模型内存。"""
import subprocess,sys

if __name__=='__main__':
    for arguments in [['tools.e33gc_g2b_train','train'],['tools.e33gc_g2b_eval']]:
        print('PYTHON_MODULE',*arguments,flush=True)
        subprocess.run([sys.executable,'-m',*arguments],check=True)
