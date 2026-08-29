from app.frontend.utils.state_utils import feature_selection_fingerprint, confirmation_freshness, is_runtime_ready

def test_fingerprint_is_stable_and_order_insensitive():
    a=[{"id":"a","name":" Python ","category":"Skills","source_type":"explicit_skill"},{"id":"b","name":"Git","category":"Skills"}]
    assert feature_selection_fingerprint(a)==feature_selection_fingerprint(list(reversed(a)))
    assert feature_selection_fingerprint(a)!=feature_selection_fingerprint(a[:1])

def test_freshness_lifecycle():
    state={"run_id":"r","document_id":"d","runtime_parsed":{"id":"d"},"confirm_status":"APPLIED","confirmed_feature_fingerprint":feature_selection_fingerprint([{"id":"a","name":"A","category":"Skills"}])}
    assert confirmation_freshness({}, [])=="UNCONFIRMED"
    assert confirmation_freshness(state,[{"id":"a","name":"A","category":"Skills"}])=="CONFIRMED"
    assert confirmation_freshness(state,[{"id":"b","name":"B","category":"Skills"}])=="DIRTY"
    assert is_runtime_ready(state,[{"id":"a","name":"A","category":"Skills"}])
    assert not is_runtime_ready(state,[{"id":"b","name":"B","category":"Skills"}])
