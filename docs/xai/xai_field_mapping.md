# XAI field mapping

| XAI field | Existing source |
|---|---|
| `jd_id`, `cv_id`, `job_title` | normalized JD/CV IDs and `jd.role.job_title` |
| `decision.final_score`, `status`, `coverage` | `mdms` result |
| `decision.weights` | frozen MDMS weights |
| `dimensions.skill` | `skill_matcher.match_skills` |
| `dimensions.experience.years` | `experience_matcher` years result |
| `dimensions.experience.evidence` | responsibility evidence items and CV experience chunks |
| `dimensions.education.degree/field` | `education_matcher` |
| `dimensions.semantic` | `semantic_matcher` profile similarity/status |
| `evidence_registry` | normalized CV/JD source artifacts and selected matcher evidence |
| `strength_candidates`, `gap_candidates`, `interview_focus` | deterministic projection of matcher decisions |

The current matching artifacts do not consistently expose stable evidence IDs or per-dimension weighted contributions; C2 should add those mappings without changing scoring.
