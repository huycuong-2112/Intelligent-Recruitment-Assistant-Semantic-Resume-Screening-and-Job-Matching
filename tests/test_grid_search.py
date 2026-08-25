from src.Optimization.grid_search import search, simplex_grid

def test_simplex_grid_and_search_are_deterministic():
    grid=simplex_grid(); assert len(grid)==286; assert all(abs(sum(weights)-1)<1e-12 and min(weights)>=0 for weights in grid)
    rows=[{"gt_overall":3,"skill_score":1,"experience_score":0,"education_score":0,"semantic_score":0},{"gt_overall":0,"skill_score":0,"experience_score":1,"education_score":0,"semantic_score":0}]
    first,best=search(rows); second,_=search(rows); assert first==second; assert best["w_skill"]==1.0
