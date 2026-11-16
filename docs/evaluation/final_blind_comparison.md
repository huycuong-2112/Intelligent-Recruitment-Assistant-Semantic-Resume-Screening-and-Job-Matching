# Final Frozen V1 Evaluation

Primary test set: `blind_all` (17 candidates), pooled from `blind_v1` (6) and `blind_v2` (11). Relevant means final human `overall >= 2`; there are 3 relevant candidates.

| Metric | Rule-Based | MDMS Equal | MDMS Tuned |
|---|---:|---:|---:|
| Recall@5 | 0.6667 | 1.0000 | 1.0000 |
| Recall@10 | 0.6667 | 1.0000 | 1.0000 |
| Recall@15 | 0.6667 | 1.0000 | 1.0000 |
| nDCG@5 | 0.8772 | 0.9675 | 0.9631 |
| nDCG@10 | 0.8460 | 1.0000 | 1.0000 |
| nDCG@15 | 0.8335 | 0.9394 | 0.9532 |
| Spearman | 0.4124 | 0.5977 | 0.6179 |
| MAE | 0.9565 | 0.7734 | 0.6676 |

MDMS Equal uses 0.25/0.25/0.25/0.25. MDMS Tuned uses 0.40/0.20/0.10/0.30. Both use identical frozen component scores. The tuned configuration is retained for release; it does not win every individual metric.

Limitations: one JD, IT-only data, N=17 pooled blind candidates, and only three relevant candidates. Results are project-level evidence, not population-level ATS performance claims. Metrics where K exceeds the split size are N/A.

Scoring and evaluation policy were frozen before blind labels were opened; no post-blind retuning was performed.
