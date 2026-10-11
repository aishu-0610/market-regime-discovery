import pytest

from app import create_app


@pytest.fixture(scope="module")
def client():
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.get_json() == {"status": "ok"}


def test_features_all(client):
    r = client.get("/api/features/SPY")
    assert r.status_code == 200
    body = r.get_json()
    assert body["rows"] == 5221
    assert len(body["dates"]) == body["rows"]
    assert len(body["columns"]) == 7
    for col in body["columns"]:
        assert len(body["data"][col]) == body["rows"]


def test_features_splits_do_not_overlap(client):
    train = client.get("/api/features/SPY?split=train").get_json()
    test = client.get("/api/features/SPY?split=test").get_json()
    assert train["dates"][-1] < test["dates"][0]
    assert train["rows"] == 3021
    assert test["rows"] == 2200


def test_features_lowercase_ticker_works(client):
    assert client.get("/api/features/spy?split=test").status_code == 200


def test_bad_split_is_400(client):
    r = client.get("/api/features/SPY?split=bogus")
    assert r.status_code == 400
    assert "error" in r.get_json()


def test_unknown_ticker_is_404(client):
    r = client.get("/api/features/NOPE")
    assert r.status_code == 404
    assert "error" in r.get_json()


def test_summary_shape(client):
    r = client.get("/api/features/SPY/summary")
    assert r.status_code == 200
    body = r.get_json()
    assert body["split"]["train_rows"] == 3021
    assert body["split"]["test_rows"] == 2200
    assert body["scaler"]["fit_on"] == "train only"
    assert set(body["train"].keys()) == set(body["test"].keys())
    assert abs(body["train"]["ret_1d"]["count"] - 3021) < 1e-9