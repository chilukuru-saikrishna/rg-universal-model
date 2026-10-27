"""Player data: simulated players with all 89 RG data points, or real data from a file."""
import numpy as np
import pandas as pd

from .config import COLS, RANDOM_STATE


def simulate_players(n_players: int = 8000, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Generate synthetic players with the same 89 columns and patterns as the Excel workbook.

    A hidden risk score drives every behaviour (with noise), and a simulated RG outcome is
    derived from it. FOR TESTING ONLY - the patterns are written into this generator.
    """
    N = n_players
    rng = np.random.default_rng(seed)
    r = rng.normal(0, 1, N)          # hidden risk
    pos = np.maximum(r, 0)

    def cl(x, a, b):
        return np.clip(x, a, b)

    def n(s=1):
        return rng.normal(0, s, N)

    d = {}
    d[1]=cl(np.round(42-6*r+n(10)),18,80).astype(int)
    d[2]=np.where(rng.random(N)<0.6+0.05*r,"Male","Female")
    d[3]=rng.choice(["UK","Spain","Sweden","Denmark","Malta","Ontario"],N,p=[.35,.2,.15,.1,.1,.1])
    d[4]=rng.integers(5,1500,N)
    d[5]=np.where(rng.random(N)<0.3+0.1*pos,"Low",np.where(rng.random(N)<0.6,"Medium","High"))
    d[6]=np.where(rng.random(N)<0.06+0.08*(r>1),"Yes","No")
    act=cl(np.round(8+4*r+n(4)),1,30); d[7]=act.astype(int)
    sess=np.round(act*cl(1.5+0.4*r+n(.4),1,5)); d[8]=sess.astype(int)
    avg=cl(35-5*r+n(12),4,120); d[9]=np.round(avg,1)
    mx=np.round(avg*cl(2+0.6*pos+n(.4),1.2,6)); d[10]=mx.astype(int)
    d[11]=np.round(sess*avg/60,1)
    night=cl(.12+.08*r+n(.07),0,.9); d[12]=np.round(night,2)
    d[13]=np.round(cl(.3+n(.1),0,1),2)
    d[14]=np.round(cl(.15+.05*r+n(.06),0,1),2)
    d[15]=np.round(mx*cl(.8+.1*pos+n(.1),.5,1)).astype(int)
    d[16]=np.minimum(act,cl(np.round(3+2.5*r+n(2)),1,30)).astype(int)
    d[17]=np.round(cl(np.abs(n(1))+.6*pos,0,8),1)
    depc=np.round(cl(sess*(1.1+.5*r+n(.4)),1,None)); d[18]=depc.astype(int)
    avgdep=cl(40+15*r+n(20),10,None); totdep=np.round(depc*avgdep); d[19]=totdep.astype(int)
    d[20]=np.round(depc/np.maximum(sess,1),2)
    d[21]=cl(np.round(1+1.3*pos+n(.8)),1,12).astype(int)
    d[22]=cl(np.round(1*pos+np.abs(n(.8))+(r>.5)*1.5),0,20).astype(int)
    d[23]=np.round(720/np.maximum(depc,1)*cl(1+n(.2),.5,2),1)
    d[24]=np.round(cl(10+30*r+n(30),-80,400)).astype(int)
    d[25]=cl(np.round(.6*pos+np.abs(n(.7))-.3),0,15).astype(int)
    d[26]=np.round(cl(night+n(.05),0,1),2)
    d[27]=np.round(cl(.4+.2*r+n(.2),0,1),2)
    wc=cl(np.round(3-1*r+n(1.5)),0,20); d[28]=wc.astype(int)
    d[29]=np.round(cl(60-10*r+n(40),0,None)).astype(int)
    d[30]=cl(np.round(.8*pos+np.abs(n(.5))-.4),0,10).astype(int)
    wdr=cl(.6-.2*r+n(.15),0,1.5); d[31]=np.round(wdr,2)
    d[32]=cl(np.round(.6*pos+np.abs(n(.6))-.3),0,10).astype(int)
    bets=np.round(sess*cl(40+20*r+n(15),5,None)); d[33]=bets.astype(int)
    stake=np.round(bets*cl(1.5+.6*r+n(.6),.2,None)); d[34]=stake.astype(int)
    d[35]=np.round(stake/np.maximum(bets,1),2)
    d[36]=np.round(d[35]*cl(6+3*pos+n(2),2,None)).astype(int)
    d[37]=np.round(cl(.6+.25*r+n(.15),.1,3),2)
    d[38]=np.round(cl(2+6*r+n(5),-30,60),1)
    d[39]=np.round(cl(4+1.5*r+n(1),.5,20),1)
    d[40]=np.round(cl(1+.25*r+n(.15),.5,3),2)
    loss=np.round(totdep*(1-wdr*cl(.8+n(.1),.3,1))); loss=np.maximum(loss,-200); d[41]=loss.astype(int)
    d[42]=np.round(loss/np.maximum(act,1)).astype(int)
    d[43]=np.round(loss/np.maximum(stake,1),3)
    d[44]=np.round(cl(5+25*r+n(25),-80,300)).astype(int)
    dep=cl(.15+.15*r+n(.1),0,1); d[45]=np.round(dep,2)
    d[46]=np.round(np.maximum(loss,0)*cl(.3+.1*pos+n(.1),.1,1)).astype(int)
    d[47]=np.round(np.maximum(loss,0)*d[4]/30*cl(.5+n(.2),.1,2)).astype(int)
    inc={"Low":1500,"Medium":3000,"High":6000}
    d[48]=np.round(np.maximum(loss,0)/np.array([inc[x] for x in d[5]]),3)
    d[49]=cl(np.round(1.3+.6*r+n(.7)),1,5).astype(int)
    d[50]=np.round(cl(.5+.15*r+n(.2),0,1),2)
    d[51]=np.round(cl(2.8+.5*r+n(.6),1,5),1)
    d[52]=np.round(cl(2+1*r+n(1),1,15),1)
    d[53]=np.round(cl(.2+.1*r+n(.15),0,1),2)
    d[54]=np.where(rng.random(N)<.1+.15*(r>.8),"Yes","No")
    d[55]=np.where(rng.random(N)<.5,"Yes","No")
    d[56]=cl(np.round(.7*pos+np.abs(n(.5))-.3),0,10).astype(int)
    d[57]=cl(np.round(.5*pos+np.abs(n(.4))-.4),0,5).astype(int)
    d[58]=cl(np.round(np.abs(n(.6))-.2),0,5).astype(int)
    d[59]=cl(np.round(.6*pos+np.abs(n(.5))-.4),0,6).astype(int)
    se=cl(np.round(.7*pos+np.abs(n(.4))-.6),0,5).astype(int); d[60]=se
    d[61]=np.where(se>0,rng.integers(1,200,N),"").astype(object)
    d[62]=np.round(cl(6-1.5*r+n(2),.5,30),1)
    d[63]=cl(np.round(.8*pos+np.abs(n(.8))-.3),0,15).astype(int)
    d[64]=np.where((r>1.4)&(rng.random(N)<.4),"Yes","No")
    d[65]=cl(np.round(1.5+.6*r+n(.6)),1,8).astype(int)
    d[66]=cl(np.round(.5*pos+np.abs(n(.5))-.3),0,6).astype(int)
    d[67]=cl(np.round(1.3+.5*r+n(.6)),0,8).astype(int)
    d[68]=rng.choice(["Card","E-wallet","Bank transfer","Prepaid"],N,p=[.5,.3,.15,.05])
    d[69]=np.where(rng.random(N)<.03+.08*(r>1),"Yes","No")
    d[70]=np.round(cl(.3+.12*r+n(.15),0,1),2)
    d[71]=cl(np.round(.6*pos+np.abs(n(.5))-.4),0,10).astype(int)
    d[72]=np.round(cl(.2+.1*r+n(.1),0,1),2)
    d[73]=np.round(cl(.25+.1*r+n(.12),0,1),2)
    d[74]=cl(np.round(1.5+1.2*r+n(1.2)),0,30).astype(int)
    d[75]=cl(np.round(.5*pos+np.abs(n(.5))-.3),0,10).astype(int)
    d[76]=np.where(r>1.2,rng.integers(0,4,N),np.where(rng.random(N)<.03,1,0)).astype(int)
    d[77]=np.where((r>1.1)&(rng.random(N)<.35),"Yes","No")
    d[78]=np.where(r>.6,rng.choice(["Improved","No change","Worse"],N,p=[.4,.35,.25]),np.where(rng.random(N)<.1,"Improved","None"))
    d[78]=np.where((r<=.6)&(d[78]=="Improved"),"Improved",d[78])
    pg=np.where(rng.random(N)<.3,np.round(cl(4+4*r+n(3),0,27)),np.nan); d[79]=np.array(["" if np.isnan(x) else int(x) for x in pg],dtype=object)
    d[80]=np.round(cl(1.5+.8*r+n(.6),0,20),1)
    d[81]=np.round(cl(.15+.08*r+n(.08),0,1),2)
    d[82]=np.round(cl(.6+.08*r+n(.15),0,1),2)
    d[83]=cl(np.round(.4*pos+np.abs(n(.5))-.3),0,8).astype(int)
    d[84]=cl(np.round(1+.5*pos+n(.4)),1,5).astype(int)
    d[85]=np.round(cl(10+30*r+n(25),-80,300)).astype(int)
    d[86]=np.round(cl(1+5*r+n(4),-30,50),1)
    d[87]=np.round(stake*cl(.9+.3*r+n(.2),.2,3)).astype(int)
    d[88]=np.minimum(6,np.maximum(1,d[4]//30)).astype(int)
    d[89]=np.round(cl(.4+.2*r+n(.12),.05,3),2)
    out = pd.DataFrame({COLS[i - 1]: d[i] for i in range(1, 90)})
    out.insert(0, "Player ID", [f"P{k + 1:05d}" for k in range(N)])
    out["outcome"] = ((r + rng.normal(0, .7, N)) > 0.8).astype(int)
    return out


def load_real(path: str) -> pd.DataFrame:
    """Load approved, anonymised player data with the same columns (+ 'outcome')."""
    df = pd.read_excel(path) if path.endswith((".xlsx", ".xls")) else pd.read_csv(path)
    missing = [c for c in COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{len(missing)} RG columns missing, e.g. {missing[:3]}")
    if "outcome" not in df.columns:
        raise ValueError("Real data needs an 'outcome' column (1 = RG case, 0 = not)")
    return df


def check_excel(path: str) -> int:
    """Return how many RG-xxx columns the Excel 'Player Data' sheet has (expect 89)."""
    ids = pd.read_excel(path, sheet_name="Player Data", header=None, nrows=4).iloc[3].dropna()
    return sum(str(v).startswith("RG-") for v in ids)
