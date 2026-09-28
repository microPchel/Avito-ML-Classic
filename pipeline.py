"""Avito AntiBot: leakage-checked raw-data pipeline and fixed 20-model rank ensemble.
Run: python pipeline.py --data /path/to/raw --output outputs
The data folder may contain data/train.csv, data/test.csv, and either
 data/events.csv.gz or events_part1.csv.gz ... events_part3.csv.gz.
"""
from pathlib import Path
import sys,json,hashlib,warnings,argparse,time
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score,roc_auc_score
from lightgbm import LGBMClassifier
from metric import precision_at_recall
import features as sf
from features import build_pointer_features, POINTER_COLUMNS
from submission import validate_submission
warnings.filterwarnings('ignore',category=pd.errors.PerformanceWarning)
warnings.filterwarnings('ignore',category=FutureWarning)
SEEDS=[0,1,7,42,123,11,29,73,2026,3407]
EVENT_TYPES=sorted(['contact_chat_open','contact_message_sent','contact_phone_show','favorite_add','item_view','login','photo_swipe','search_results_view','seller_page_view'])
BASE_PARAMS=dict(objective='binary',n_estimators=500,learning_rate=.03,num_leaves=31,min_child_samples=30,subsample=.9,colsample_bytree=.9,reg_lambda=1.,n_jobs=4,verbosity=-1)
# subsample_freq is intentionally 0 for the supplied reference-style baseline.
POINTER_PARAMS=dict(objective='binary',colsample_bytree=0.94500090035736,extra_trees=False,learning_rate=0.027749622019760254,max_bin=255,max_depth=4,min_child_samples=20,min_split_gain=0.35989052708910574,n_estimators=1000,num_leaves=10,path_smooth=0.9087876737984324,reg_alpha=0.005269080909714222,reg_lambda=0.4508762463821835,scale_pos_weight=0.7421219676315555,subsample=0.98096822550102,subsample_freq=1,n_jobs=4,verbosity=-1)
def read_data(root):
 root=Path(root)
 dates=['cookie_created_at','window_start_ts','window_end_ts']
 tr=pd.read_csv(root/'data/train.csv',parse_dates=dates);te=pd.read_csv(root/'data/test.csv',parse_dates=dates)
 parts=[root/f'events_part{i}.csv.gz' for i in [1,2,3]]
 data_parts=[root/'data'/f'events_part{i}.csv.gz' for i in [1,2,3]]
 if (root/'data/events.csv.gz').exists():paths=[root/'data/events.csv.gz']
 elif (root/'events.csv.gz').exists():paths=[root/'events.csv.gz']
 elif all(p.exists() for p in data_parts):paths=data_parts
 elif all(p.exists() for p in parts):paths=parts
 else:raise FileNotFoundError('Provide all 3 event parts or data/events.csv.gz')
 events=pd.concat([pd.read_csv(p,parse_dates=['event_ts']) for p in paths],ignore_index=True)
 return tr,te,events

def preprocess(train,test,events):
 rawcols=list(events.columns);meta=pd.concat([train.drop(columns='target'),test],ignore_index=True)
 assert meta.cookie_id.is_unique and meta.cookie_id.notna().all()
 assert set(train.target.unique())<={0,1}
 assert meta[['window_start_ts','window_end_ts','cookie_created_at']].notna().all().all()
 assert ((meta.window_end_ts-meta.window_start_ts).dt.total_seconds()==86400).all()
 assert (meta.cookie_created_at<=meta.window_start_ts).all()
 assert events.event_ts.notna().all()
 joined=events.merge(meta[['cookie_id','window_start_ts','window_end_ts']],on='cookie_id',how='left',validate='many_to_one')
 assert joined.window_start_ts.notna().all()
 mask=joined.event_ts.ge(joined.window_start_ts)&joined.event_ts.lt(joined.window_end_ts)
 inside=joined.loc[mask,rawcols].copy()
 audit={'raw_events':len(events),'outside_events':int((~mask).sum()),'captcha_inside':int(inside.event_name.eq('captcha_shown').sum())}
 # Explicitly exclude this type even if a future data version places it inside a window.
 inside=inside.loc[inside.event_name.ne('captcha_shown')].copy()
 dup=inside.duplicated(subset=rawcols,keep='first')
 dupstats=inside.assign(_duplicate=dup).groupby('cookie_id')._duplicate.mean().rename('exact_duplicate_rate').reset_index()
 clean=inside.loc[~dup].sort_values(['cookie_id','event_ts'],kind='mergesort').reset_index(drop=True)
 audit.update(exact_duplicate_rows=int(dup.sum()),clean_events=len(clean),same_second_rows_preserved=int(clean.duplicated(['cookie_id','event_ts'],keep=False).sum()))
 assert not clean.duplicated(rawcols).any()
 assert (clean.groupby('cookie_id').event_ts.diff().dropna().dt.total_seconds()>=0).all()
 checked=clean.merge(meta[['cookie_id','window_start_ts','window_end_ts']],on='cookie_id',validate='many_to_one')
 assert (checked.event_ts.ge(checked.window_start_ts)&checked.event_ts.lt(checked.window_end_ts)).all()
 assert 'captcha_shown' not in clean.event_name.unique()
 assert set(clean.event_name.unique())<=set(EVENT_TYPES)
 sf.EVENT_TYPES=EVENT_TYPES
 return clean,dupstats,audit

def make_features(clean,meta,dupstats):
 # No target is passed to a feature generator. Aggregation is local to one cookie.
 cols=['cookie_id','cookie_created_at','window_start_ts','window_end_ts'];meta=meta[cols].copy()
 ev=clean.loc[clean.cookie_id.isin(meta.cookie_id)]
 base=sf.generate_source_features(sf.enrich_event_level(ev),meta,dupstats).set_index('cookie_id').reindex(meta.cookie_id)
 base=base.reindex(sorted(base.columns),axis=1)
 pointer=build_pointer_features(ev,meta)
 # Freeze column order independently of which pointer events are missing in test.
 pointer=pointer.reindex(columns=POINTER_COLUMNS)
 rich=base.join(pointer)
 result={k:v.replace([np.inf,-np.inf],np.nan).fillna(0).astype(float) for k,v in [('safe',base),('pointer',rich)]}
 for x in result.values():
  assert x.index.equals(pd.Index(meta.cookie_id,name='cookie_id'))
  assert not any(c in x for c in ['target','cookie_id','window_start_ts','window_end_ts'])
  assert not any('captcha' in c for c in x)
  assert np.isfinite(x.to_numpy()).all()
 return result

def metrics(y,p):
 assert np.isfinite(p).all()
 return dict(par=precision_at_recall(y,p,recall=.70),ap=average_precision_score(y,p),auc=roc_auc_score(y,p))

def to_rank(p):
 # Rank each component over the entire scoring batch. Never rank separately per day.
 return rankdata(p,method='average')  # half-integer ranks: exact before final division

def validate(features,meta,output,seeds=SEEDS):
 output=Path(output);output.mkdir(parents=True,exist_ok=True)
 y=meta.target.to_numpy(dtype=int);valid=meta.window_start_ts.ge('2026-04-17').to_numpy()
 assert int(valid.sum())==1951 and int(y[valid].sum())==160
 assert int((~valid).sum())==9140 and int(y[~valid].sum())==739
 pred={};rows=[]
 for name,params in [('safe',BASE_PARAMS),('pointer',POINTER_PARAMS)]:
  x=features[name];pred[name]=[]
  for seed in seeds:
   m=LGBMClassifier(**params,random_state=seed);m.fit(x.loc[~valid],y[~valid]);p=m.predict_proba(x.loc[valid])[:,1]
   pred[name].append(p);r=dict(model=name,seed=seed,**metrics(y[valid],p));rows.append(r);print(r,flush=True)
  pred[name]=np.stack(pred[name])
 ranks={k:np.stack([to_rank(p) for p in v]) for k,v in pred.items()}
 rank_sums=ranks['safe']+ranks['pointer']
 blend=rank_sums/(2*int(valid.sum()))
 for seed,p in zip(seeds,blend):rows.append(dict(model='rank_blend',seed=seed,**metrics(y[valid],p)))
 frame=pd.DataFrame(rows);frame.to_csv(output/'robustness.csv',index=False)
 summary=frame.groupby('model').agg(mean=('par','mean'),median=('par','median'),worst=('par','min'),best=('par','max'),std=('par','std'),ap=('ap','mean'))
 final=rank_sums.sum(axis=0)/(2*len(seeds)*int(valid.sum()));result=metrics(y[valid],final)
 for name in pred:summary.loc[name,'seed_ensemble_par']=metrics(y[valid],pred[name].mean(0))['par']
 summary.loc['rank_blend','seed_ensemble_par']=result['par'];summary.to_csv(output/'robustness_summary.csv')
 result['first_5_seeds']=metrics(y[valid],rank_sums[:5].sum(0)/(2*min(5,len(seeds))*int(valid.sum())))['par'];result['last_5_seeds']=metrics(y[valid],rank_sums[5:].sum(0)/(2*(len(seeds)-5)*int(valid.sum())))['par'] if len(seeds)>5 else None
 result['seeds']=list(seeds);result['selection_note']='Fixed 50/50 rank blend selected on this temporal holdout; no independent labeled test estimate.'
 (output/'validation_metrics.json').write_text(json.dumps(result,indent=2))
 v=meta.loc[valid,['cookie_id','window_start_ts','target']].copy();v['score']=final
 for name,pp in pred.items():
  for seed,p in zip(seeds,pp):v[f'{name}_{seed}']=p
 v.to_csv(output/'validation_predictions.csv',index=False)
 print(summary,flush=True);print('FINAL VALIDATION',result,flush=True)
 return result,v

def fit_full(features,test_features,train,test,output,seeds=SEEDS):
 output=Path(output);output.mkdir(parents=True,exist_ok=True);scores=[]
 for name,params in [('safe',BASE_PARAMS),('pointer',POINTER_PARAMS)]:
  x=features[name];xt=test_features[name].reindex(columns=x.columns)
  for seed in seeds:
   m=LGBMClassifier(**params,random_state=seed);m.fit(x,train.target.to_numpy(dtype=int));p=m.predict_proba(xt)[:,1];scores.append(to_rank(p))
   
   print('full train:',name,seed,flush=True)
 submission=pd.DataFrame({'cookie_id':test.cookie_id.to_numpy(),'score':np.sum(scores,axis=0)/(len(scores)*len(test))})
 validate_submission(submission,test.cookie_id)
 submission.to_csv(output/'submission.csv',index=False)

 return submission

def main(data_root,output,validate_only=False):
 output=Path(output);output.mkdir(parents=True,exist_ok=True)
 train,test,events=read_data(data_root);clean,dup,audit=preprocess(train,test,events)
 print('AUDIT',audit,flush=True);(output/'data_audit.json').write_text(json.dumps(audit,indent=2))
 feats=make_features(clean,train,dup);test_feats=make_features(clean,test,dup)
 for name,x in feats.items():
  (output/f'features_{name}.json').write_text(json.dumps(list(x.columns),indent=2))
 result,pred=validate(feats,train,output)
 if not validate_only:fit_full(feats,test_feats,train,test,output)
 return result
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--data',default='.');parser.add_argument('--output',default='outputs');parser.add_argument('--validate-only',action='store_true');args=parser.parse_args();main(args.data,args.output,args.validate_only)
