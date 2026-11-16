"""Materialize frozen three-system evaluation artifacts from existing V1 outputs."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error
from src.Evaluation.metrics import ndcg_at_k

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'Data/Results/IT/FinalEvaluation'; OUT.mkdir(parents=True,exist_ok=True)
gt=pd.read_excel(ROOT/'Data/GroundTruth/IT/groundtruth_overall_annotation_v2-2.xlsx'); gt=gt.rename(columns={'split':'source_split'})
gt['split']=gt.source_split.replace({'blind_test':'blind_v1'}); gt['relevant']=gt.overall>=2
gt.to_csv(OUT/'final_gt_validated.csv',index=False)
rows=[]
for p in sorted((ROOT/'Data/Results/IT/Matching/jd_001').glob('cv_*.json')):
 d=json.loads(p.read_text(encoding='utf-8')); c=d.get('components',{}); rows.append({'cv_id':d['cv_id'],'jd_id':d['jd_id'],'S_skill':c.get('skill',d.get('skill',{}).get('score',0.0)),'S_experience':c.get('experience',d.get('experience',{}).get('score',0.0)),'S_education':c.get('education',d.get('education',{}).get('score',0.0)),'S_semantic':c.get('semantic',d.get('semantic',{}).get('score',0.0))})
base=pd.DataFrame(rows).merge(gt[['pair_id','cv_id','jd_id','split','overall','relevant']],on=['cv_id','jd_id'],how='inner').rename(columns={'overall':'human_overall'})
weights={'mdms_equal_v1':{'S_skill':.25,'S_experience':.25,'S_education':.25,'S_semantic':.25},'mdms_tuned_v1':{'S_skill':.4,'S_experience':.2,'S_education':.1,'S_semantic':.3}}
for name,w in weights.items():
 x=base.copy(); x['score_0_1']=sum(x[k]*v for k,v in w.items()); x['score_0_3']=x.score_0_1*3; x.to_csv(OUT/f'{name}_predictions_all35.csv',index=False)
rb=[]
for p in sorted((ROOT/'Data/Results/IT/Baselines/rule_based/jd_001').glob('cv_*.json')):
 d=json.loads(p.read_text(encoding='utf-8')); rb.append({'cv_id':d['cv_id'],'jd_id':d['jd_id'],'score_0_1':d.get('score_0_1',0.0)})
rb=base.drop(columns=['S_skill','S_experience','S_education','S_semantic','score_0_1'] if 'score_0_1' in base else []).merge(pd.DataFrame(rb),on=['cv_id','jd_id']).copy(); rb['score_0_3']=rb.score_0_1*3; rb.to_csv(OUT/'rule_baseline_predictions_all35.csv',index=False)
def metrics(x):
 x=x.sort_values('score_0_1',ascending=False); n=len(x); rel=int(x.relevant.sum()); out={'N':n,'relevant_count':rel}
 for k in (5,10,15):
  out[f'recall_at_{k}']=None if k>n else float(x.head(k).relevant.sum()/rel) if rel else None
  out[f'ndcg_at_{k}']=None if k>n else float(ndcg_at_k(x.head(k).human_overall.tolist(),k=k,warnings=[]))
 out['spearman']=float(spearmanr(x.human_overall,x.score_0_3).statistic) if n>1 else None; out['mae']=float(mean_absolute_error(x.human_overall,x.score_0_3)); return out
allrows=[]
for system,x in [('mdms_equal_v1',pd.read_csv(OUT/'mdms_equal_v1_predictions_all35.csv')),('mdms_tuned_v1',pd.read_csv(OUT/'mdms_tuned_v1_predictions_all35.csv')),('rule_based_baseline',pd.read_csv(OUT/'rule_baseline_predictions_all35.csv'))]:
 for split in ('development','blind_v1','blind_v2','blind_all','full_35'):
  y=x if split=='full_35' else x[x.split.isin(['blind_v1','blind_v2'])] if split=='blind_all' else x[x.split==split]
  m=metrics(y); m.update(system=system,split=split,evaluation_type='DESCRIPTIVE' if split=='full_35' else ('DEVELOPMENT' if split=='development' else 'BLIND')); allrows.append(m)
pd.DataFrame(allrows).to_csv(OUT/'evaluation_metrics_by_split.csv',index=False)
for split in ('development','blind_v1','blind_v2','blind_all'):
 x=pd.read_csv(OUT/'mdms_tuned_v1_predictions_all35.csv'); x=x if split=='development' else x[x.split.isin(['blind_v1','blind_v2'])] if split=='blind_all' else x[x.split==split]; x=x.sort_values('score_0_1',ascending=False).copy(); x.insert(0,'rank',range(1,len(x)+1)); x.to_csv(OUT/f'{split}_ranking.csv',index=False)
print('FINAL_EVALUATION_ARTIFACTS_OK',len(base))
