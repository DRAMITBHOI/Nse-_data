import os, glob, json
from datetime import datetime
import numpy as np
import pandas as pd

DATA_DIR = "data"
NIFTY750_FILE = os.path.join(DATA_DIR, "nifty750.json")
START_DATE, OOS_SPLIT_DATE = "2021-01-01", "2024-01-01"
MIN_DELIV_COVERAGE = 0.80
H25, H60 = 25, 60

EXCLUDED = {"nifty750.json", "nifty50.json", "nifty.json",
            "fundamentals.json", "accuracy_audit_log.json"}
BAD_TOKENS = ("report", "results", "scanner", "watchlist", "leaderboard",
              "sweep", "audit", "backtest", "champion", "history", "nifty")


def is_non_stock_file(name):
    low = name.lower()
    return low in EXCLUDED or any(x in low for x in BAD_TOKENS)


def number(x):
    if x is None or str(x).strip() == "":
        return np.nan
    try:
        y = float(x)
        return y if np.isfinite(y) else np.nan
    except Exception:
        return np.nan


def load_nifty750_symbols():
    if not os.path.exists(NIFTY750_FILE):
        raise RuntimeError(f"Missing {NIFTY750_FILE}")
    with open(NIFTY750_FILE, encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, list):
        items = raw
    elif isinstance(raw, dict) and isinstance(raw.get("symbols"), list):
        items = raw["symbols"]
    elif isinstance(raw, dict) and isinstance(raw.get("constituents"), list):
        items = raw["constituents"]
    elif isinstance(raw, dict):
        items = raw.keys()
    else:
        raise RuntimeError("Unsupported nifty750.json schema")
    symbols = {str(x).strip().upper() for x in items if str(x).strip()}
    if len(symbols) < 400:
        raise RuntimeError(f"Only {len(symbols)} symbols found; expected at least 400.")
    print(f"Loaded {len(symbols)} symbols from {NIFTY750_FILE}.")
    return symbols


def quality(row):
    return (int(row["delivery_observed"]), int(np.isfinite(row["deliv_pct_raw"])),
            int(np.isfinite(row["delivery_vol"]) and row["delivery_vol"] > 0),
            row["volume"])


def clean_stock(raw, sym, exclusions, duplicates, splits, audit):
    if not isinstance(raw, list):
        exclusions.append((sym, "not_a_list", 0))
        return None
    by_date = {}
    for r in raw:
        if not isinstance(r, dict):
            continue
        d = str(r.get("time", ""))[:10]
        if len(d) != 10:
            continue
        vals = [number(r.get(k)) for k in ("open", "high", "low", "close", "volume")]
        if not np.all(np.isfinite(vals)):
            continue
        o, h, l, c, v = vals
        if min(vals) <= 0 or h < max(o, c)-0.01 or l > min(o, c)+0.01 or h < l:
            continue
        dv, dp = number(r.get("delivery_vol")), number(r.get("deliv_pct"))
        obs = int(np.isfinite(dv) and dv >= 0)
        if r.get("delivery_vol") not in (None, "") and not obs:
            audit["malformed_delivery_fields"] += 1
        if not np.isfinite(dp) or not 0 <= dp <= 100:
            if r.get("deliv_pct") not in (None, ""):
                audit["malformed_delivery_fields"] += 1
            dp = np.nan
        row = dict(date=d, open=o, high=h, low=l, close=c, volume=v,
                   delivery_vol=dv if obs else np.nan, deliv_pct_raw=dp,
                   delivery_observed=obs)
        if d not in by_date:
            by_date[d] = row
        else:
            old = by_date[d]
            ohlc_diff = max(abs(old[k]-row[k]) for k in ("open","high","low","close"))
            dv_diff = not ((np.isnan(old["delivery_vol"]) and np.isnan(row["delivery_vol"]))
                           or old["delivery_vol"] == row["delivery_vol"])
            dp_diff = not ((np.isnan(old["deliv_pct_raw"]) and np.isnan(row["deliv_pct_raw"]))
                           or (np.isfinite(old["deliv_pct_raw"]) and np.isfinite(row["deliv_pct_raw"])
                               and abs(old["deliv_pct_raw"]-row["deliv_pct_raw"]) <= .05))
            if ohlc_diff > .05 or old["volume"] != v or dv_diff or dp_diff:
                keep, drop = (row, old) if quality(row) >= quality(old) else (old, row)
                duplicates.append((sym,d,round(ohlc_diff,3),keep["volume"],drop["volume"],
                                   keep["delivery_vol"],drop["delivery_vol"]))
                by_date[d] = keep

    rows = [by_date[d] for d in sorted(by_date)]
    if len(rows) < 260:
        exclusions.append((sym, "insufficient_rows_<260", len(rows)))
        return None

    # Corporate-action heuristic; inspect audit output manually.
    multipliers = [2., 3., 4., 5., 10.]
    for i in range(len(rows)-1, 0, -1):
        prev_c, curr_o = rows[i-1]["close"], rows[i]["open"]
        ratio = prev_c/curr_o if curr_o > 0 else 0
        jump = rows[i]["volume"]/rows[i-1]["volume"] if rows[i-1]["volume"] > 0 else 1
        adj = next((m for m in multipliers if ratio >= 1.75 and jump >= 1.25
                    and abs(ratio-m)/m < .12), None)
        if adj:
            splits.append((sym, rows[i]["date"], round(prev_c,2), round(curr_o,2),
                           round(ratio,3), round(jump,2), adj))
            for j in range(i):
                for k in ("open","high","low","close"):
                    rows[j][k] /= adj
                rows[j]["volume"] *= adj
                if rows[j]["delivery_observed"]:
                    rows[j]["delivery_vol"] *= adj

    df = pd.DataFrame(rows)
    audit["total"] += len(df)
    audit["missing"] += int((df.delivery_observed == 0).sum())
    audit["zero"] += int(((df.delivery_observed == 1) & (df.delivery_vol == 0)).sum())
    audit["positive"] += int(((df.delivery_observed == 1) & (df.delivery_vol > 0)).sum())
    audit["delivery_gt_volume"] += int(
        ((df.delivery_observed == 1) & (df.delivery_vol > df.volume)).sum())
    positive = df[(df.delivery_observed == 1) & (df.delivery_vol > 0)]
    if len(positive) < 120:
        exclusions.append((sym, "low_observed_delivery_days_<120", len(positive)))
        return None
    ratio = positive.delivery_vol / positive.volume
    if ratio.round(4).nunique() < 15:
        exclusions.append((sym, "flat_delivery_ratios_<15", ratio.round(4).nunique()))
        return None

    df["deliv_vol_clean"] = np.where(
        df.delivery_observed == 1, np.minimum(df.delivery_vol, df.volume), np.nan)
    df["deliv_pct_clean"] = np.where(
        df.delivery_observed == 1, df.deliv_vol_clean / df.volume * 100, np.nan)
    return df


def add_indicators(df):
    df = df.copy()
    df["vol_sma20"] = df.volume.shift(1).rolling(20, min_periods=20).mean()
    obs = df.delivery_observed.shift(1)
    df["deliv_cov20"] = obs.rolling(20, min_periods=20).mean()
    df["deliv_sma20"] = (df.deliv_vol_clean.shift(1).rolling(20, min_periods=1).mean()
                         .where(df.deliv_cov20 >= MIN_DELIV_COVERAGE))
    dpobs = df.deliv_pct_clean.notna().astype(int).shift(1).rolling(50, min_periods=50).mean()
    prior_dp = df.deliv_pct_clean.shift(1)
    df["dpct_cov50"] = dpobs
    df["dpct_mean50"] = prior_dp.rolling(50, min_periods=1).mean().where(dpobs >= MIN_DELIV_COVERAGE)
    df["dpct_std50"] = prior_dp.rolling(50, min_periods=1).std().where(
        dpobs >= MIN_DELIV_COVERAGE).replace(0, np.nan)
    df["dpct_z"] = (df.deliv_pct_clean-df.dpct_mean50)/df.dpct_std50
    df["to_cr"] = df.close*df.volume/1e7
    df["to_50d"] = df.to_cr.shift(1).rolling(50, min_periods=50).mean()
    df["sma50"] = df.close.rolling(50, min_periods=50).mean()
    df["sma200"] = df.close.rolling(200, min_periods=200).mean()
    df["high_252"] = df.high.shift(1).rolling(252, min_periods=252).max()
    df["perf_60d"] = df.close.pct_change(60)
    df["perf_30d"] = df.close.pct_change(30)
    surge = df.deliv_vol_clean/df.deliv_sma20.replace(0,np.nan)
    absret = df.close.pct_change().abs().mul(100).clip(lower=.25)
    df["daily_absorption"] = surge/absret
    direction = np.sign(df.close.diff())
    df["deliv_obv"] = (direction*df.deliv_vol_clean.fillna(0)).cumsum()
    df["dobv_20max"] = df.deliv_obv.rolling(20,min_periods=20).max()
    df["obv_cov_21d"] = df.delivery_observed.rolling(21,min_periods=21).mean()
    pc = df.close.shift(1)
    tr = pd.concat([df.high-df.low,(df.high-pc).abs(),(df.low-pc).abs()],axis=1).max(axis=1)
    df["atr14"] = tr.shift(1).rolling(14,min_periods=14).mean()
    df["atr5"] = tr.shift(1).rolling(5,min_periods=5).mean()
    rng = df.high-df.low
    df["rc"] = np.where(rng>0,(df.close-df.low)/rng,.5)
    return df


def load_nifty():
    for fname in sorted(os.listdir(DATA_DIR)):
        if fname.lower() in {"nifty.json", "nifty50.json"}:
            p = os.path.join(DATA_DIR, fname)
            with open(p, encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, dict):
                raw = raw.get("data") or raw.get("candles") or []
            dm = {}
            for r in raw:
                if isinstance(r, dict) and r.get("time"):
                    c = number(r.get("close"))
                    if np.isfinite(c) and c > 0:
                        dm[str(r["time"])[:10]] = c
            if len(dm) >= 200:
                s = pd.Series(dm).sort_index()
                perf = s.pct_change(60)
                sma = s.rolling(50, min_periods=50).mean()
                valid = perf.notna() & sma.notna()
                print(f"Loaded Nifty benchmark from {fname}: {len(dm)} days ({s.index.min()} -> {s.index.max()}).")
                return perf[valid].to_dict(), (s[valid] >= sma[valid]).astype(int).to_dict()
    raise RuntimeError("Nifty benchmark requires >=200 valid daily observations.")


def benchmark_on_date(date, perf, regime, dates):
    if date in perf and date in regime: return perf[date],regime[date]
    prior=[x for x in dates if x<=date]
    if not prior: return np.nan,np.nan
    last=prior[-1]
    age=(datetime.strptime(date,"%Y-%m-%d")-datetime.strptime(last,"%Y-%m-%d")).days
    return (perf[last],regime[last]) if age<=5 else (np.nan,np.nan)


def eval_forward(hi,lo,cl,start,entry,stop,complete60):
    risk=entry-stop
    if risk<=0 or start>=len(cl):
        return ("INVALID_RISK",0.,np.nan,0.,0.,"INVALID_RISK")
    target=entry+1.5*risk
    end25=min(len(cl),start+H25)
    outcome,realised,mfe,mae="TIMEOUT_25D",None,0.,0.
    for k in range(start,end25):
        mfe=max(mfe,(hi[k]-entry)/risk)
        mae=min(mae,(lo[k]-entry)/risk)
        hs,ht=lo[k]<=stop,hi[k]>=target
        if hs and ht: outcome,realised="AMBIGUOUS",0.; break
        if hs: outcome,realised="STOP_FIRST",-1.; break
        if ht: outcome,realised="SUCCESS_25D",1.5; break
    close_r=(cl[end25-1]-entry)/risk
    if realised is None: realised=close_r
    out60=outcome
    if outcome=="TIMEOUT_25D":
        if not complete60: out60="INCOMPLETE_60D"
        else:
            out60="TIMEOUT_60D"
            for k in range(end25,min(len(cl),start+H60)):
                hs,ht=lo[k]<=stop,hi[k]>=target
                if hs and ht: out60="AMBIGUOUS_60D"; break
                if hs: out60="STOP_60D"; break
                if ht: out60="SUCCESS_60D"; break
    return outcome,round(float(realised),3),round(float(close_r),3),round(mfe,3),round(mae,3),out60


def eval_entry(hi,lo,cl,entry_idx,entry,stop,min_risk=.01,max_risk=.15):
    if entry_idx>=len(cl): return ("UNFILLED_ENTRY",0.,np.nan,0.,0.,"UNFILLED_ENTRY",np.nan)
    if entry<=stop or stop<=0 or entry<=0:
        return ("INVALID_RISK",0.,np.nan,0.,0.,"INVALID_RISK",np.nan)
    riskpct=(entry-stop)/entry
    if not min_risk<=riskpct<=max_risk:
        return ("INVALID_RISK",0.,np.nan,0.,0.,"INVALID_RISK",riskpct)
    if not lo[entry_idx]<=entry<=hi[entry_idx]:
        return ("UNFILLED_ENTRY",0.,np.nan,0.,0.,"UNFILLED_ENTRY",riskpct)
    target=entry+1.5*(entry-stop)
    if lo[entry_idx]<=stop or hi[entry_idx]>=target:
        return ("AMBIGUOUS_ENTRY_DAY",0.,np.nan,0.,0.,"AMBIGUOUS_ENTRY_DAY",riskpct)
    complete60=int(entry_idx+H60<len(cl))
    result=eval_forward(hi,lo,cl,entry_idx+1,entry,stop,complete60)
    return (*result,riskpct)


def summary(title,df,outcome,rcol,mfecol=None):
    if df.empty:
        print(f"{title}: N=0"); return
    labels=["SUCCESS_25D","STOP_FIRST","TIMEOUT_25D","AMBIGUOUS"]
    x=df[df[outcome].isin(labels)]
    if x.empty:
        print(f"{title}: no classified filled trades"); return
    vc=x[outcome].value_counts(normalize=True)*100
    resolved=x[x[outcome].isin(labels[:3])]
    pos=resolved.loc[resolved[rcol]>0,rcol].sum()
    neg=abs(resolved.loc[resolved[rcol]<0,rcol].sum())
    pf=pos/neg if neg>0 else np.nan
    mfe=resolved[mfecol].median() if mfecol and len(resolved) else np.nan
    print(f"{title} | N={len(x)} | unfilled={(df[outcome]=='UNFILLED_ENTRY').sum()} "
          f"| entry-day ambiguous={(df[outcome]=='AMBIGUOUS_ENTRY_DAY').sum()}")
    print("Success={:.1f}% Stop={:.1f}% Timeout={:.1f}% Ambiguous={:.1f}% "
          "MeanR={:+.3f} PF={} MedianMFE={}".format(
          vc.get("SUCCESS_25D",0),vc.get("STOP_FIRST",0),vc.get("TIMEOUT_25D",0),
          vc.get("AMBIGUOUS",0),resolved[rcol].mean() if len(resolved) else np.nan,
          f"{pf:.2f}" if np.isfinite(pf) else "N/A",
          f"{mfe:.2f}" if np.isfinite(mfe) else "N/A"))


def run_discovery():
    os.makedirs(DATA_DIR,exist_ok=True)
    n750=load_nifty750_symbols()
    nifty_perf,nifty_reg=load_nifty()
    nifty_dates=sorted(nifty_perf)
    files=[p for p in sorted(glob.glob(os.path.join(DATA_DIR,"*.json")))
           if not is_non_stock_file(os.path.basename(p))]
    exclusions,duplicates,splits=[],[],[]
    audit=dict(total=0,missing=0,zero=0,positive=0,delivery_gt_volume=0,malformed_delivery_fields=0)
    stocks={}
    turnover,breadth_all,breadth_n750=[],[],[]

    # Pass 1: clean and calculate indicators.
    for path in files:
        sym=os.path.splitext(os.path.basename(path))[0].upper()
        try:
            with open(path,encoding="utf-8") as f: raw=json.load(f)
            df=clean_stock(raw,sym,exclusions,duplicates,splits,audit)
            if df is None: continue
            df=add_indicators(df); stocks[sym]=df
            sub=df.loc[df.date>=START_DATE,["date","to_50d","close","sma50"]].dropna()
            for r in sub.itertuples(index=False):
                turnover.append((r.date,sym,r.to_50d))
                above=int(r.close>=r.sma50)
                breadth_all.append((r.date,above))
                if sym in n750: breadth_n750.append((r.date,above))
        except Exception as e:
            exclusions.append((sym,f"error:{str(e)[:100]}",0))

    pd.DataFrame(exclusions,columns=["symbol","reason","detail"]).to_csv(
        os.path.join(DATA_DIR,"audit_exclusions.csv"),index=False)
    pd.DataFrame(duplicates,columns=["symbol","date","ohlc_diff","kept_volume","dropped_volume",
        "kept_delivery","dropped_delivery"]).to_csv(
        os.path.join(DATA_DIR,"audit_duplicate_dates.csv"),index=False)
    pd.DataFrame(splits,columns=["symbol","date","prev_close","curr_open","ratio","vol_jump","multiple"]).to_csv(
        os.path.join(DATA_DIR,"audit_corporate_actions.csv"),index=False)

    hist_top={}
    if turnover:
        t=pd.DataFrame(turnover,columns=["date","symbol","turnover"])
        t["rank"]=t.groupby("date").turnover.rank(ascending=False,method="min")
        hist_top={d:set(g.symbol) for d,g in t[t["rank"]<=750].groupby("date")}
    def bmap(rows):
        if not rows:return {}
        x=pd.DataFrame(rows,columns=["date","above"])
        return (x.groupby("date").above.mean()*100).round(2).to_dict()
    ba,bn=bmap(breadth_all),bmap(breadth_n750)
    total=max(audit["total"],1)
    print(f"Prepared {len(stocks)} stocks; exclusions={len(exclusions)}; duplicates={len(duplicates)}; splits={len(splits)}")
    print(f"Delivery coverage audit: missing={audit['missing']/total:.1%}, observed zero={audit['zero']/total:.1%}, "
          f"positive={audit['positive']/total:.1%}, delivery>volume={audit['delivery_gt_volume']}, "
          f"malformed fields={audit['malformed_delivery_fields']}")

    events,baseline=[],[]
    for sym,df in stocks.items():
        curr750=int(sym in n750)
        arr={c:df[c].to_numpy() for c in df.columns}
        date=arr["date"]; op,hi,lo,cl,vol=[arr[x] for x in ("open","high","low","close","volume")]
        vs,ds,dv,obs=[arr[x] for x in ("vol_sma20","deliv_sma20","deliv_vol_clean","delivery_observed")]
        dp,dpmean,dpz=[arr[x] for x in ("deliv_pct_clean","dpct_mean50","dpct_z")]
        to,sma200,p60,p30,atr14,atr5,rc=[arr[x] for x in
            ("to_50d","sma200","perf_60d","perf_30d","atr14","atr5","rc")]
        h252,absorb=[arr[x] for x in ("high_252","daily_absorption")]
        dobv,dobvmax,obvcov=[arr[x] for x in ("deliv_obv","dobv_20max","obv_cov_21d")]

        # 252-bar warm-up; reserve next-day entry plus 25 forward sessions.
        for i in range(252,len(df)-26):
            d=date[i]
            if d<START_DATE: continue
            req=[vs[i],sma200[i],to[i],p60[i],p30[i],atr14[i]]
            if not np.all(np.isfinite(req)) or vs[i]<=0 or atr14[i]<=0: continue
            nr,nreg=benchmark_on_date(d,nifty_perf,nifty_reg,nifty_dates)
            if not np.isfinite(nr) or not np.isfinite(nreg): continue
            if not (obs[i]==1 and np.isfinite(dv[i]) and np.isfinite(ds[i]) and ds[i]>0): continue
            if cl[i]<.90*sma200[i]: continue
            hist750=int(sym in hist_top.get(d,set()))
            min_to=5.0 if curr750 or hist750 else 2.0
            if to[i]<min_to or cl[i]<=op[i] or vol[i]<1.05*vs[i]: continue
            shelves=[]
            for n in range(7,26):
                sh,sl=np.max(hi[i-n:i]),np.min(lo[i-n:i])
                if sl>0 and (sh-sl)/sl<=.22 and cl[i]>sh:
                    shelves.append((n,float(sh),float(sl),(sh-sl)/sl))
            if not shelves: continue

            surge=dv[i]/ds[i]; vsurge=vol[i]/vs[i]
            rs=p60[i]-nr; normrs=rs/(atr14[i]/cl[i])
            c1=surge*max(.1,p60[i]/nr) if nr>0 else np.nan
            # Preserve the historical baseline fallback when Nifty 60d return <= 0.
            orig_score=surge*max(.1,p60[i]/nr if nr>0 else 1.0+abs(p60[i]))
            c2=surge*max(.1,1+rs); c2u=surge*(1+rs)
            vdu3=np.min(vol[i-3:i])/vs[i]; vdu5=np.min(vol[i-5:i])/vs[i]
            atrcontract=atr5[i]/atr14[i]
            dist52=cl[i]/h252[i] if np.isfinite(h252[i]) and h252[i]>0 else np.nan
            event=f"{sym}_{d}"

            # Faithful baseline shelf selection: first width-qualified 10-18 day shelf.
            original=None; maxwidth=.08 if curr750 else .13
            for n in range(10,19):
                sh,sl=np.max(hi[i-n:i]),np.min(lo[i-n:i])
                if sl>0 and (sh-sl)/sl<=maxwidth:
                    original=(n,float(sh),float(sl),(sh-sl)/sl); break
            if original:
                n,sh,sl,sp=original
                start=i-15
                coverage=np.mean(obs[start:i]==1)
                up=sum(dv[k] for k in range(start,i) if obs[k]==1 and np.isfinite(dv[k]) and cl[k]>=cl[k-1])
                down=sum(dv[k] for k in range(start,i) if obs[k]==1 and np.isfinite(dv[k]) and cl[k]<cl[k-1])
                skew=up/down if down>0 else (np.inf if up>0 else np.nan)
                reqskew=1.5 if curr750 else 1.15
                dvolok=dv[i]>=1.25*ds[i]
                dpctok=np.isfinite(dp[i]) and np.isfinite(dpmean[i]) and dp[i]>=1.25*dpmean[i]
                obvok=True
                if curr750:
                    obvok=(i>0 and np.isfinite(dobvmax[i-1]) and np.isfinite(obvcov[i])
                           and obvcov[i]==1 and dobv[i]>=dobvmax[i-1])
                stop=round(min(lo[i],sl),2); rdist=sh-stop
                score=orig_score
                orig_ok=(coverage>=MIN_DELIV_COVERAGE and skew>=reqskew and cl[i]>=sma200[i]
                    and p60[i]>nr and cl[i]>=sh and vsurge>=1.4 and (dvolok or dpctok)
                    and rc[i]>=.65 and obvok and (not curr750 or (np.isfinite(score) and score>=2.5))
                    and rdist/sh>.01 and rdist/sh<=.12 and to[i]>=10)
                if orig_ok:
                    z=eval_entry(hi,lo,cl,i+1,sh,stop,.01,.12)
                    baseline.append(dict(event_id=event,symbol=sym,date=d,is_oos=int(d>=OOS_SPLIT_DATE),
                        is_n750_current=curr750,base_len=n,spread_pct=sp*100,entry=sh,stop=stop,
                        conviction=score,delivery_trigger="BOTH" if dvolok and dpctok else
                        "DELIVERY_VOLUME" if dvolok else "DELIVERY_PERCENT",
                        risk_pct=z[6]*100 if np.isfinite(z[6]) else np.nan,
                        outcome_25d=z[0],realized_r=z[1],mfe_r=z[3],mae_r=z[4]))

            for shelf_idx,(n,pivot,sl,spread) in enumerate(shelves):
                entry=round(pivot,2); stop=round(min(lo[i],sl)*.995,2)
                a=eval_entry(hi,lo,cl,i+1,entry,stop)
                up=sum(dv[k] for k in range(i-n,i) if obs[k]==1 and np.isfinite(dv[k]) and cl[k]>=cl[k-1])
                down=sum(dv[k] for k in range(i-n,i) if obs[k]==1 and np.isfinite(dv[k]) and cl[k]<cl[k-1])
                cov=np.mean(obs[i-n:i]==1)
                skew=up/down if down>0 else (10. if up>0 else np.nan)
                if cov<MIN_DELIV_COVERAGE: skew=np.nan
                aw=absorb[i-n:i]; acov=np.isfinite(aw).mean()
                baseabs=np.nanmean(aw) if acov>=MIN_DELIV_COVERAGE else np.nan
                c3=surge*np.sqrt(min(skew,5))*max(.1,1+normrs/10) if np.isfinite(skew) else np.nan

                # B/C entries: exact fill on signal day i; exit evaluation begins i+1.
                cheat=int((hi[i-1]-lo[i-1] <= .75*atr14[i]) and
                          (hi[i-1]>=sl+.65*(pivot-sl)) and (hi[i-1]<pivot))
                b=("NO_CHEAT",0.,np.nan,0.,0.,"NO_CHEAT",np.nan)
                c=("NO_CHEAT",0.,np.nan,0.,0.,"NO_CHEAT",np.nan)
                if cheat:
                    eb=round(hi[i-1]+.05,2)
                    sb=round(lo[i-1]*.995,2)
                    b=eval_entry(hi,lo,cl,i,eb,sb)
                    sc=round(eb-1.5*atr14[i],2)
                    c=eval_entry(hi,lo,cl,i,eb,sc)

                events.append(dict(event_id=event,symbol=sym,date=d,next_date=date[i+1],
                    is_oos=int(d>=OOS_SPLIT_DATE),is_n750_current=curr750,
                    is_hist_top750_liq=hist750,is_shortest_shelf=int(shelf_idx==0),
                    nifty_60d_return=nr,nifty_above_50=int(nreg),
                    breadth_all_50sma=ba.get(d,np.nan),breadth_n750_50sma=bn.get(d,np.nan),
                    to_50d_cr=to[i],base_len=n,base_delivery_coverage=cov,absorption_coverage=acov,
                    spread_pct=spread*100,spread_atr=(pivot-sl)/atr14[i],
                    atr_contraction=atrcontract,vdu_3d_ratio=vdu3,vdu_5d_ratio=vdu5,
                    deliv_skew=skew,base_absorption=baseabs,
                    prior_30d_pct=p30[i-n]*100,dist_52w_high=dist52,
                    rs_60d_diff=rs*100,norm_rs=normrs,conv_c1_original=c1,
                    conv_c2_floored=c2,conv_c2_unfloored=c2u,conv_c3_orthogonal=c3,
                    vol_surge=vsurge,deliv_surge=surge,deliv_pct_z=dpz[i],candle_rc=rc[i],
                    risk_a_pct=a[6]*100 if np.isfinite(a[6]) else np.nan,
                    outcome_a_25d=a[0],realized_r_a=a[1],close_r_a=a[2],
                    mfe_r_a=a[3],mae_r_a=a[4],outcome_a_60d=a[5],
                    is_cheat_setup=cheat,risk_b_pct=b[6]*100 if np.isfinite(b[6]) else np.nan,
                    outcome_b_25d=b[0],realized_r_b=b[1],
                    risk_c_pct=c[6]*100 if np.isfinite(c[6]) else np.nan,
                    outcome_c_25d=c[0],realized_r_c=c[1]))

    res=pd.DataFrame(events); base=pd.DataFrame(baseline)
    if res.empty: raise RuntimeError("No discovery events found; inspect inputs and filters.")
    res.to_csv(os.path.join(DATA_DIR,"launchpad_discovery_corrected.csv"),index=False)
    base.to_csv(os.path.join(DATA_DIR,"launchpad_original_baseline_corrected.csv"),index=False)
    print(f"Saved {len(res)} shelf rows / {res.event_id.nunique()} unique breakouts.")
    print(f"Saved {len(base)} original-baseline events.")
    exp=res[res.is_shortest_shelf==1].copy()
    summary("A1 shortest shelf ALL",exp,"outcome_a_25d","realized_r_a","mfe_r_a")
    summary("A1 IS",exp[exp.is_oos==0],"outcome_a_25d","realized_r_a","mfe_r_a")
    summary("A1 OOS",exp[exp.is_oos==1],"outcome_a_25d","realized_r_a","mfe_r_a")
    if not base.empty:
        summary("Original baseline ALL",base,"outcome_25d","realized_r","mfe_r")
        summary("Original baseline IS",base[base.is_oos==0],"outcome_25d","realized_r","mfe_r")
        summary("Original baseline OOS",base[base.is_oos==1],"outcome_25d","realized_r","mfe_r")

    # Univariate bands: derive on IS only, validate on OOS. No fallback "optimal" band.
    features=["base_len","spread_pct","spread_atr","atr_contraction","vdu_3d_ratio","vdu_5d_ratio",
        "deliv_skew","base_absorption","prior_30d_pct","dist_52w_high","norm_rs",
        "conv_c1_original","conv_c2_floored","conv_c2_unfloored","conv_c3_orthogonal",
        "vol_surge","deliv_surge","deliv_pct_z","candle_rc","risk_a_pct"]
    fit_labels=["SUCCESS_25D","STOP_FIRST","TIMEOUT_25D"]
    for name,grp in [("CURRENT_NIFTY750",exp[exp.is_n750_current==1]),
                     ("HISTORICAL_TOP750_LIQUIDITY_PROXY",exp[exp.is_hist_top750_liq==1]),
                     ("NON_TOP750_LIQUIDITY_PROXY",exp[exp.is_hist_top750_liq==0])]:
        ins=grp[(grp.is_oos==0)&grp.outcome_a_25d.isin(fit_labels)]
        oos=grp[(grp.is_oos==1)&grp.outcome_a_25d.isin(fit_labels+["AMBIGUOUS"])]
        print(f"\n=== FEATURE BANDS: {name} ===")
        for f in features:
            x=ins[f].replace([np.inf,-np.inf],np.nan).dropna()
            wins=ins.loc[ins.outcome_a_25d=="SUCCESS_25D",f].dropna()
            losses=ins.loc[ins.outcome_a_25d=="STOP_FIRST",f].dropna()
            if len(x)<30 or len(wins)<20 or len(losses)<20:
                print(f"{f}: insufficient sample"); continue
            try:
                _,edges=pd.qcut(x,q=10,retbins=True,duplicates="drop")
                edges=np.unique(edges); best=-np.inf; band=None
                min_n=max(30,int(len(x)*.15))
                for j in range(len(edges)-1):
                    for k in range(j+1,len(edges)):
                        lo_b,hi_b=edges[j],edges[k]
                        sub=ins[ins[f].between(lo_b,hi_b)]
                        if len(sub)<min_n: continue
                        score=(sub.outcome_a_25d=="SUCCESS_25D").mean()*100 - \
                              .5*(sub.outcome_a_25d=="STOP_FIRST").mean()*100
                        if score>best: best,band=score,(lo_b,hi_b)
                if band is None:
                    print(f"{f}: NO VALID BAND"); continue
                lo_b,hi_b=band
                si=ins[ins[f].between(lo_b,hi_b)]
                so=oos[oos[f].between(lo_b,hi_b)]
                sr=so[so.outcome_a_25d.isin(fit_labels)]
                pos=sr.loc[sr.realized_r_a>0,"realized_r_a"].sum()
                neg=abs(sr.loc[sr.realized_r_a<0,"realized_r_a"].sum())
                pf=pos/neg if neg>0 else np.nan
                oos_n = len(so)
                oos_success = (so.outcome_a_25d == "SUCCESS_25D").mean() * 100 if oos_n else np.nan
                oos_stop = (so.outcome_a_25d == "STOP_FIRST").mean() * 100 if oos_n else np.nan
                oos_timeout = (so.outcome_a_25d == "TIMEOUT_25D").mean() * 100 if oos_n else np.nan
                oos_ambiguous = (so.outcome_a_25d == "AMBIGUOUS").mean() * 100 if oos_n else np.nan
                median_mfe = sr.mfe_r_a.median() if len(sr) else np.nan
                pf_text = f"{pf:.2f}" if np.isfinite(pf) else "N/A"
                print(
                    f"{f:<20} {lo_b:.4g}..{hi_b:.4g} | IS N={len(si)} | OOS N={oos_n} "
                    f"| Success={oos_success:.1f}% | Stop={oos_stop:.1f}% "
                    f"| Timeout={oos_timeout:.1f}% | Ambiguous={oos_ambiguous:.1f}% "
                    f"| MeanR={sr.realized_r_a.mean() if len(sr) else np.nan:+.3f} "
                    f"| PF={pf_text} | MedianMFE={median_mfe:.2f}"
                    if np.isfinite(median_mfe) else
                    f"{f:<20} {lo_b:.4g}..{hi_b:.4g} | IS N={len(si)} | OOS N={oos_n} "
                    f"| Success={oos_success:.1f}% | Stop={oos_stop:.1f}% "
                    f"| Timeout={oos_timeout:.1f}% | Ambiguous={oos_ambiguous:.1f}% "
                    f"| MeanR={sr.realized_r_a.mean() if len(sr) else np.nan:+.3f} "
                    f"| PF={pf_text} | MedianMFE=N/A"
                )
            except Exception as e:
                print(f"{f}: band error: {e}")

    # Shelf duration distribution: rows are correlated variants, not independent trades.
    res["duration_bucket"]=pd.cut(res.base_len,[6,9,13,18,25],
        labels=["7-9d","10-13d","14-18d","19-25d"])
    print("\n=== DURATION BUCKETS (SHELF VARIANTS; NOT INDEPENDENT EVENTS) ===")
    for bucket,g in res.groupby("duration_bucket",observed=False):
        if len(g):
            summary(f"{bucket}: rows={len(g)}, unique events={g.event_id.nunique()}",
                    g,"outcome_a_25d","realized_r_a","mfe_r_a")

    cheat=exp[exp.is_cheat_setup==1]
    print(f"\n=== CHEAT-ELIGIBLE SETUP N={len(cheat)} ===")
    for label,out,rcol in [("A1 pivot","outcome_a_25d","realized_r_a"),
                           ("B tight cheat","outcome_b_25d","realized_r_b"),
                           ("C ATR cheat","outcome_c_25d","realized_r_c")]:
        summary(label,cheat,out,rcol)


if __name__ == "__main__":
    run_discovery()
