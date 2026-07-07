from core.logging_utils import _resolve_model_perf_profile, log_model_inference_metrics


def test_resolve_model_perf_profile_crowd():
    task, model, size = _resolve_model_perf_profile("helmet_detection", detector_name="crowd")
    assert task == "人员聚集检测"
    assert model == "Person + 聚类分析"
    assert size == "640×640"


def test_resolve_model_perf_profile_fall_input_size():
    task, model, size = _resolve_model_perf_profile(
        "fall_detection",
        model_path="models/fall-26s-320-best_int8_9b.bmodel",
    )
    assert task == "摔倒检测"
    assert size == "320×320"


def test_log_model_inference_metrics_smoke(tmp_path, monkeypatch):
    import logging

    log_path = tmp_path / "bm-web-test.log"
    monkeypatch.setenv("BM_WEB_LOG_FILE", str(log_path))
    from core.logging_utils import configure_metrics_logging, log_model_inference_metrics

    configure_metrics_logging(force=True)
    logging.getLogger("bm_metrics").handlers[0].flush = lambda: None

    log_model_inference_metrics(
        0,
        "fight_detection",
        timings={
            "preprocess_ms": 180.0,
            "inference_ms": 20.0,
            "postprocess_ms": 0.5,
            "total_ms": 200.5,
        },
        detector_name="fight",
    )
    text = log_path.read_text(encoding="utf-8")
    assert "端到端/ms=200.50" in text
    assert "预处理/ms=180.00" in text
