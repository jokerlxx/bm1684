from app.interfaces.http.app_factory import create_app


def test_http_app_keeps_status_route_compatible():
    app = create_app(
        handlers={
            "status": lambda: {"system_running": False, "detectors": {}, "streams": []},
            "start": lambda: {"status": "success"},
            "stop": lambda: {"status": "success"},
            "toggle_detector": lambda data: {"status": "success", "data": data},
        },
        get_scheduler=lambda: None,
        get_alert_queue=lambda: None,
    )

    client = app.test_client()
    response = client.get("/api/status")

    assert response.status_code == 200
    assert response.get_json()["system_running"] is False
