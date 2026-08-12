import numpy as np
import json
import faiss
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def encode_text(texts):
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    return embeddings / norms

def build_index(cv_list):
    if not cv_list:
        return faiss.IndexFlatIP(384)
    
    vectors = encode_text([cv['text'] for cv in cv_list])
    index = faiss.IndexFlatIP(384)
    index.add(vectors.astype(np.float32))
    return index

# def rank_candidates(jd_text, cv_list, index, top_k=10):
#     if not cv_list or index.ntotal == 0:
#         return []
        
#     jd_vector = encode_text([jd_text]).astype(np.float32)
#     search_k = max(1, min(top_k, len(cv_list)))
    
#     scores, indices = index.search(jd_vector, search_k)
    
#     results = []
#     for score, idx in zip(scores[0], indices[0]):
#         if idx != -1 and idx < len(cv_list):
#             results.append((cv_list[idx], float(score)))
            
#     return results

def get_scores_for_metrics(jd_text, cv_list, index):
    jd_vector = encode_text([jd_text]).astype(np.float32)
    scores, indices = index.search(jd_vector, len(cv_list))
    
    y_score = np.zeros(len(cv_list))
    for score, idx in zip(scores, indices):
        if idx != -1:
            y_score[idx] = float(score)
    return y_score

with open("../../Data/Processed/cleaned_resumes.json", 'r', encoding='utf-8') as f:
    res_list = json.load(f)

jd_path = ""
