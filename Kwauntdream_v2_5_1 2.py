import os, json, numpy as np, pandas as pd, matplotlib.pyplot as plt
from itertools import combinations
from numpy.linalg import eigh, norm

np.set_printoptions(precision=4, suppress=True)

def laplacian_erdos(n:int, p=0.05, seed=None):
    rng = np.random.default_rng(seed)
    A = (rng.random((n,n))<p).astype(float)
    A = np.triu(A,1); A = A + A.T
    np.fill_diagonal(A, 0)
    D = np.diag(A.sum(1))
    return D - A, A

def laplacian_smallworld(n:int, k=4, beta=0.2, seed=None):
    rng = np.random.default_rng(seed)
    A = np.zeros((n,n), float)
    for i in range(n):
        for s in range(1, k//2+1):
            j=(i+s)%n; jj=(i-s)%n
            A[i,j]=A[j,i]=1; A[i,jj]=A[jj,i]=1
    for i in range(n):
        for s in range(1, k//2+1):
            j=(i+s)%n
            if rng.random()<beta:
                A[i,j]=A[j,i]=0
                choices = np.where(A[i]==0)[0]
                choices = choices[choices!=i]
                if choices.size>0:
                    t = rng.choice(choices)
                    A[i,t]=A[t,i]=1
    D = np.diag(A.sum(1))
    return D - A, A

def quantum_walk(L, steps=12, tau=0.05, gamma=1.0, init_idx=0):
    n=L.shape[0]
    psi=np.zeros(n,complex); psi[init_idx]=1.0
    for _ in range(steps):
        psi = psi - 1j*tau*gamma*(L@psi)
        psi /= norm(psi)
    return psi

def snapshot_p_phi(psi, n_qubits:int, clip=(0.05,0.95)):
    amp = psi[:n_qubits]
    p = np.abs(amp)**2
    p = p / (p.max() + 1e-12)
    p = np.clip(p, clip[0], clip[1])
    phi = np.angle(amp)
    return p, phi

def rho2_pseudo_exact(p_i, phi_i, p_j, phi_j, alpha=0.25):
    w = np.array([(1-p_i)*(1-p_j),(1-p_i)*p_j,p_i*(1-p_j),p_i*p_j], float)
    phases = np.array([0.0, phi_j, phi_i, phi_i+phi_j], float)
    rho = np.zeros((4,4), complex)
    for a in range(4):
        for b in range(4):
            hd = ((a&1)!=(b&1)) + (((a>>1)&1)!=((b>>1)&1))
            rho[a,b] = np.exp(-alpha*hd) * np.exp(1j*(phases[a]-phases[b])) * np.sqrt(w[a]*w[b])
    rho = (rho + rho.conj().T)/2
    return rho/ rho.trace().real

_X = np.array([[0,1],[1,0]], complex)
_Y = np.array([[0,-1j],[1j,0]], complex)
_Z = np.array([[1,0],[0,-1]], complex)
_PAULIS = [_X,_Y,_Z]
_KRONS = [np.kron(a,b) for a in _PAULIS for b in _PAULIS]

def chsh_max_horodecki(rho2):
    T = np.array([np.trace(rho2 @ K).real for K in _KRONS], float).reshape(3,3)
    vals,_ = eigh(T.T @ T)
    m1,m2 = np.sort(vals)[-2:]
    return float(2*np.sqrt(max(0.0, m1+m2)))

def negativity(rho):
    rhoPT = rho.reshape(2,2,2,2).transpose(0,3,2,1).reshape(4,4)
    ev = np.linalg.eigvalsh(rhoPT)
    return float(np.sum(np.maximum(0.0, -ev)))

def U_theta(theta):
    U=np.eye(4, dtype=complex); U[3,3]=np.exp(1j*theta)
    return U

def ent_map_and_nodes(graph_type="smallworld", n_nodes=64, pseudo_count=24, alpha=0.25, seed=0, theta_points=11):
    if graph_type=="smallworld":
        L,_ = laplacian_smallworld(n_nodes, k=4, beta=0.2, seed=seed)
    elif graph_type=="erdos":
        L,_ = laplacian_erdos(n_nodes, p=0.05, seed=seed)
    else:
        raise ValueError("graph_type must be 'smallworld' or 'erdos'")
    psi=quantum_walk(L, steps=12, tau=0.05, gamma=1.0, init_idx=0)
    p,phi=snapshot_p_phi(psi, pseudo_count)
    theta_grid=np.linspace(0, np.pi, theta_points)
    Nmat=np.zeros((pseudo_count,pseudo_count), float)
    for i,j in combinations(range(pseudo_count),2):
        base=rho2_pseudo_exact(p[i],phi[i],p[j],phi[j], alpha=alpha)
        best=base; best_ch=chsh_max_horodecki(base)
        for th in theta_grid:
            r = U_theta(th) @ base @ U_theta(th).conj().T
            ch = chsh_max_horodecki(r)
            if ch>best_ch: best, best_ch = r, ch
        Nmat[i,j]=Nmat[j,i]=negativity(best)
    node_neg = Nmat.sum(axis=1)/(np.maximum(1,(pseudo_count-1)))
    deg = np.diag(L).astype(float)[:pseudo_count]
    evals,evecs = eigh(L)
    fied = np.abs(evecs[:, np.argsort(evals)[1]])[:pseudo_count]
    nodes = pd.DataFrame({
        "graph_type": graph_type,
        "pseudo_count": pseudo_count,
        "idx": np.arange(pseudo_count),
        "avg_neg": node_neg,
        "degree": deg,
        "fiedler_abs": fied
    })
    return nodes

def run_test_suite(outdir="/mnt/data/graphwave_v2_5_1_test"):
    os.makedirs(outdir, exist_ok=True)
    graph_types=["smallworld","erdos"]
    pseudo_counts=[24,32,40]
    seeds_per_cfg=[0,1,2,3,4]
    rows=[]
    node_tables=[]
    for gtype in graph_types:
        for pc in pseudo_counts:
            cfg_nodes=[]
            for s in seeds_per_cfg:
                nodes = ent_map_and_nodes(graph_type=gtype, pseudo_count=pc, seed=1000*s + (0 if gtype=="smallworld" else 100))
                cfg_nodes.append(nodes)
            alln = pd.concat(cfg_nodes, ignore_index=True)
            agg = alln.groupby("idx").agg(
                avg_neg_mean=("avg_neg","mean"),
                avg_neg_std=("avg_neg","std"),
                degree_mean=("degree","mean"),
                fiedler_mean=("fiedler_abs","mean")
            ).reset_index()
            agg["graph_type"]=gtype
            agg["pseudo_count"]=pc
            node_tables.append(agg)
            rows.append({
                "graph_type": gtype,
                "pseudo_count": pc,
                "mean_avg_neg_mean": float(agg["avg_neg_mean"].mean()),
                "max_avg_neg_mean": float(agg["avg_neg_mean"].max()),
                "std_avg_neg_mean": float(agg["avg_neg_mean"].std()),
            })
    df_nodes = pd.concat(node_tables, ignore_index=True)
    df_summary = pd.DataFrame(rows)
    nodes_csv=os.path.join(outdir, "Kwauntdream_nodes.csv")
    summary_csv=os.path.join(outdir, "Kwauntdream_summary.csv")
    df_nodes.to_csv(nodes_csv, index=False)
    df_summary.to_csv(summary_csv, index=False)
    for g in graph_types:
        sub = df_summary[df_summary.graph_type==g].sort_values("pseudo_count")
        plt.plot(sub["pseudo_count"], sub["mean_avg_neg_mean"], marker="o")
        plt.title(f"Kwauntdream — mean(avg_neg) vs pseudo_count ({g})")
        plt.xlabel("pseudo_count"); plt.ylabel("mean(avg_neg) over nodes")
        plt.savefig(os.path.join(outdir, f"Kwauntdream_mean_vs_pc_{g}.png"), bbox_inches="tight"); plt.close()
    return {"nodes_csv": nodes_csv, "summary_csv": summary_csv, "outdir": outdir, "summary_df": df_summary}

if __name__ == "__main__":
    artifacts = run_test_suite()
    print(json.dumps({"nodes_csv": artifacts["nodes_csv"], "summary_csv": artifacts["summary_csv"], "outdir": artifacts["outdir"]}, indent=2))
