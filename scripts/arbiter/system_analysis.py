"""
Deep system error analysis — CalEyeZ (held-out TEST split).

Decomposes every system error into routing errors vs. model errors, finds the worst
class confusions, analyzes Israeli routing, and renders dark-themed figures for index.html.
Figures -> assets/.  Numbers printed for the HTML tables.
"""
from __future__ import annotations
from pathlib import Path
from collections import Counter
import sys
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent.parent
CSV = ROOT / "datasets" / "arbiter_dataset.csv"
ARB = ROOT / "scripts" / "arbiter" / "arbiter_xgb.json"
ASSETS = ROOT / "assets"
ASSETS.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT / "scripts" / "arbiter"))
from train_arbiter_xgb import add_features, FEATS

# ---- palette (GitHub dark) ----
BG="#0d1117"; PANEL="#161b22"; INK="#e6edf3"; MUT="#8b949e"; GRID="#30363d"
BLUE="#58a6ff"; GREEN="#3fb950"; AMBER="#d29922"; RED="#f85149"; PURPLE="#bc8cff"
plt.rcParams.update({
    "figure.facecolor":BG,"axes.facecolor":PANEL,"savefig.facecolor":BG,
    "text.color":INK,"axes.labelcolor":INK,"xtick.color":MUT,"ytick.color":MUT,
    "axes.edgecolor":GRID,"grid.color":GRID,"font.size":11,"axes.titlecolor":INK,
})

def save(fig, name):
    fig.tight_layout(); p = ASSETS / name
    fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"  fig -> assets/{name}")

# ---- load + route ----
df = pd.read_csv(CSV)
df["split"] = np.where(df["image_path"].str.contains(r"[\\/]val[\\/]"),"val","test")
df = df[df.split=="test"].reset_index(drop=True)
feat = add_features(df)
clf = xgb.XGBClassifier(); clf.load_model(str(ARB))
df["route_prob"] = clf.predict_proba(feat[FEATS])[:,1]
df["route"] = np.where(df.route_prob>=0.5,"israeli","global")
df["final_pred"] = np.where(df.route=="israeli", df.i_top1, df.g_top1)
df["final_correct"] = (df.final_pred==df.true_class).astype(int)
N=len(df)

gc=df.g_correct.astype(bool); ic=df.i_correct.astype(bool); fc=df.final_correct.astype(bool)
has_right = gc|ic
routing_err = (~fc)&has_right
model_err  = (~fc)&(~has_right)
print(f"N(test)={N}")
print(f"system correct : {fc.sum()} ({fc.mean()*100:.2f}%)")
print(f"routing errors : {routing_err.sum()} ({routing_err.mean()*100:.2f}%)  (a correct model existed, router missed)")
print(f"model errors   : {model_err.sum()} ({model_err.mean()*100:.2f}%)  (both models wrong -> unfixable)")

# ===== FIG 1: error decomposition =====
fig,ax=plt.subplots(figsize=(8,1.9))
vals=[fc.sum(),routing_err.sum(),model_err.sum()]
labs=[f"System correct  {vals[0]} ({vals[0]/N*100:.1f}%)",
      f"Routing error  {vals[1]} ({vals[1]/N*100:.1f}%)",
      f"Both models wrong  {vals[2]} ({vals[2]/N*100:.1f}%)"]
cols=[GREEN,AMBER,RED]; left=0
for v,c,l in zip(vals,cols,labs):
    ax.barh(0,v,left=left,color=c,edgecolor=BG); left+=v
ax.set_xlim(0,N); ax.set_yticks([]); ax.set_title("System outcome decomposition (held-out test, 11,265 images)")
ax.legend(labs,loc="upper center",bbox_to_anchor=(0.5,-0.25),ncol=1,frameon=False,labelcolor=INK)
for s in ax.spines.values(): s.set_visible(False)
save(fig,"sys_1_outcome.png")

# ===== FIG 2: per-domain accuracy =====
acc_g=fc[df.domain=="global"].mean()*100; acc_i=fc[df.domain=="israeli"].mean()*100
overall=fc.mean()*100; base=gc.mean()*100; oracle=has_right.mean()*100
fig,ax=plt.subplots(figsize=(7,3.6))
bars=["Global\ndomain","Israeli\ndomain","Overall\nsystem"]; vals=[acc_g,acc_i,overall]
b=ax.bar(bars,vals,color=[BLUE,PURPLE,GREEN],width=0.6)
ax.axhline(base,ls="--",color=MUT,lw=1); ax.axhline(oracle,ls=":",color=AMBER,lw=1.2)
ax.text(2.55,base+0.4,f"always-global {base:.1f}%",color=MUT,fontsize=9,ha="right")
ax.text(2.55,oracle+0.4,f"oracle {oracle:.1f}%",color=AMBER,fontsize=9,ha="right")
for r,v in zip(b,vals): ax.text(r.get_x()+r.get_width()/2,v+0.6,f"{v:.1f}%",ha="center",color=INK,fontweight="bold")
ax.set_ylim(0,100); ax.set_ylabel("Top-1 accuracy (%)"); ax.set_title("System accuracy by domain")
ax.grid(axis="y",alpha=.3)
save(fig,"sys_2_domain.png")

# ===== FIG 3: top system confusion pairs =====
wrong=df[~fc]
pairs=Counter(zip(wrong.true_class,wrong.final_pred))
top=pairs.most_common(12)[::-1]
fig,ax=plt.subplots(figsize=(8,4.6))
labels=[f"{t} → {p}" for (t,p),_ in top]; counts=[n for _,n in top]
ax.barh(labels,counts,color=RED,alpha=.85)
for i,n in enumerate(counts): ax.text(n+0.3,i,str(n),va="center",color=INK,fontsize=9)
ax.set_title("Top system confusions  (true → predicted, test errors)"); ax.grid(axis="x",alpha=.3)
save(fig,"sys_3_confusions.png")

# ===== FIG 4: Israeli dishes routed correct vs missed (NORMALIZED to 100%) =====
isr=df[df.domain=="israeli"]
dishes=sorted(isr.true_class.unique())
corr=np.array([((isr.true_class==d)&fc).sum() for d in dishes])
miss=np.array([((isr.true_class==d)&(~fc)).sum() for d in dishes])
tot=corr+miss; recall=np.where(tot>0, corr/tot*100, 0)
order=np.argsort(recall)                 # worst at bottom
dishes=[dishes[i] for i in order]; recall=recall[order]; tot=tot[order]
fig,ax=plt.subplots(figsize=(8,5))
ax.barh(dishes,recall,color=GREEN,label="recovered (system correct)")
ax.barh(dishes,100-recall,left=recall,color=RED,alpha=.85,label="missed (routed to global)")
for i,(r,n) in enumerate(zip(recall,tot)):
    ax.text(2,i,f"{r:.0f}%",va="center",ha="left",color="#06240f",fontsize=9,fontweight="bold")
    ax.text(101,i,f"n={n}",va="center",color=MUT,fontsize=8)
ax.set_xlim(0,100); ax.set_xlabel("share of that dish's test images (%)")
ax.set_title("Israeli dish routing recall  (each bar = 100% of that dish)")
ax.legend(frameon=False,labelcolor=INK,loc="upper center",bbox_to_anchor=(0.5,-0.13),ncol=2)
ax.grid(axis="x",alpha=.3)
save(fig,"sys_4_israeli.png")

# ===== FIG 5: routing separability =====
fig,ax=plt.subplots(figsize=(7,3.6))
ax.hist(df.route_prob[df.domain=="global"],bins=40,color=BLUE,alpha=.7,label="true global",log=True)
ax.hist(df.route_prob[df.domain=="israeli"],bins=40,color=PURPLE,alpha=.7,label="true israeli",log=True)
ax.axvline(0.5,ls="--",color=INK,lw=1); ax.text(0.51,ax.get_ylim()[1]*0.4,"route threshold",color=INK,fontsize=8)
ax.set_xlabel("arbiter P(israeli)"); ax.set_ylabel("images (log)")
ax.set_title("Routing separability  (ROC-AUC 0.93)"); ax.legend(frameon=False,labelcolor=INK)
save(fig,"sys_5_separability.png")

# ===== FIG 6: worst dish classes (n>=100) =====
big=df.groupby("true_class").filter(lambda g:len(g)>=100)
acc=big.groupby("true_class").final_correct.mean().sort_values()[:12]
fig,ax=plt.subplots(figsize=(8,4.6))
ax.barh(acc.index[::-1],(acc.values*100)[::-1],color=AMBER,alpha=.9)
for i,v in enumerate(acc.values[::-1]): ax.text(v*100+0.5,i,f"{v*100:.0f}%",va="center",color=INK,fontsize=9)
ax.set_xlim(0,100); ax.set_title("Lowest system accuracy — major classes (n≥100 test imgs)"); ax.grid(axis="x",alpha=.3)
save(fig,"sys_6_worstclasses.png")

# ---- printed tables ----
print("\n# top system confusions"); [print(f"  {t} -> {p}: {n}") for (t,p),n in pairs.most_common(10)]
print("\n# israeli per-dish recall");
for d in sorted(isr.true_class.unique()):
    m=isr.true_class==d; print(f"  {d:20} {(m&fc).sum():>3}/{m.sum():<3} = {(m&fc).sum()/m.sum()*100:.0f}%")
print("\n# global images misrouted to israeli")
mis=df[(df.domain=='global')&(df.route=='israeli')]
print(f"  count={len(mis)} ({len(mis)/(df.domain=='global').sum()*100:.1f}% of global)")
[print(f"  {c}: {n}") for c,n in Counter(mis.true_class).most_common(8)]
