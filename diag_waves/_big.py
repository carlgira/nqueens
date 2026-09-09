import numpy as np, time, sys
sys.setrecursionlimit(30000)
from diag_waves.prune import sample_states, simple_features
from diag_waves.multitask import MultiTask, auc
rng = np.random.default_rng(0)

def build(ns, per):
    X,y=[],[]
    for n in ns:
        for cur,ok in sample_states(n, per_depth=per, seed=n*7+1):
            X.append(simple_features(cur,n)); y.append(0. if ok else 1.)
    return np.array(X,np.float32), np.array(y,np.float32)

Xtr,ytr = build([10,11,12], 2500)
mu,sd = Xtr.mean(0), Xtr.std(0)+1e-6; Z=((Xtr-mu)/sd).astype(np.float32)
mt = MultiTask(7,(64,32),1,rng)
mt.fit(Z, np.zeros((len(Z),1),np.float32), ytr, epochs=400, wa=0.0, rng=rng)
w=np.zeros(7); b=0.
for _ in range(8000):
    p=1/(1+np.exp(-(Z@w+b))); g=(p-ytr)/len(Z); w-=.15*(Z.T@g+1e-5*w); b-=.15*g.sum()

def sample_big(n, fracs, per, cap=600000):
    full=(1<<n)-1; out=[]; nodes=[0]
    def ext(r, cols, dp, dm):
        nodes[0]+=1
        if nodes[0]>cap: raise TimeoutError
        if r==n: return True
        free=~(cols|dp|dm)&full
        while free:
            bb=free&-free; free^=bb
            if ext(r+1, cols|bb, (dp|bb)<<1, (dm|bb)>>1): return True
        return False
    to=0
    for f in fracs:
        d=int(n*f); got=0; guard=0
        while got<per and guard<per*30:
            guard+=1
            cols=dp=dm=0; cur=[]; ok=True
            for r in range(d):
                free=~(cols|dp|dm)&full
                if not free: ok=False; break
                bits=[]
                while free:
                    bb=free&-free; free^=bb; bits.append(bb)
                bb=bits[rng.integers(len(bits))]
                cur.append(bb.bit_length()-1); cols|=bb; dp=(dp|bb)<<1; dm=(dm|bb)>>1
            if not ok: continue
            nodes[0]=0
            try:
                e=ext(d, cols, dp, dm)
            except (TimeoutError, RecursionError):
                to+=1; continue
            out.append((tuple(cur), e)); got+=1
    return out, to

print('entrenado en N=10,11,12 · probado en N nunca vistos')
print(f'{"N":>5} {"estados":>8} {"% cond.":>8} {"AUC red":>9} {"AUC formula":>13} {"sin resolver":>13} {"seg":>7}')
print('-'*70)
for n, per in ((20,400),(30,300),(40,250),(50,200),(64,150),(80,120),(100,100)):
    t0=time.time()
    S, to = sample_big(n, (0.55,0.65,0.75,0.85,0.92), per)
    if not S: print(f'{n:>5}  sin muestras'); continue
    X = np.array([simple_features(c,n) for c,_ in S], np.float32)
    y = np.array([0. if ok else 1. for _,ok in S], np.float32)
    if 0 < y.mean() < 1:
        Zt=((X-mu)/sd).astype(np.float32)
        _,_,_,s = mt.fwd(Zt)
        print(f'{n:>5} {len(S):>8,} {y.mean():>7.0%} {auc(s,y):>9.3f} {auc(Zt@w+b,y):>13.3f} '
              f'{to:>13,} {time.time()-t0:>7.0f}')
    else:
        print(f'{n:>5} {len(S):>8,} {y.mean():>7.0%}   (una sola clase)')
