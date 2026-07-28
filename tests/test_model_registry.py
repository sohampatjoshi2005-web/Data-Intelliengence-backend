from app.services.model_registry import LocalModelRegistry


def build_registry(tmp_path, monkeypatch) -> LocalModelRegistry:
    monkeypatch.setenv("MODEL_REGISTRY_DIR", str(tmp_path))
    return LocalModelRegistry()


def test_rollback_rejects_current_production_target(tmp_path, monkeypatch):
    registry = build_registry(tmp_path, monkeypatch)
    registry.register({"model_id": "champion", "stage": "production"})

    result = registry.rollback_to("champion", actor="approver")

    assert result == {"ok": False, "error": "target_already_production"}
    assert registry.get_model("champion")["stage"] == "production"


def test_rollback_promotes_prior_candidate(tmp_path, monkeypatch):
    registry = build_registry(tmp_path, monkeypatch)
    registry.register({"model_id": "champion", "stage": "production"})
    registry.register({"model_id": "candidate", "stage": "staging"})

    result = registry.rollback_to("candidate", actor="approver")

    assert result == {"ok": True, "model_id": "candidate", "stage": "production"}
    assert registry.get_model("champion")["stage"] == "staging"
    candidate = registry.get_model("candidate")
    assert candidate["stage"] == "production"
    assert candidate["history"][-1]["event"] == "rollback_promote"
    assert candidate["history"][-1]["stage"] == "production"
