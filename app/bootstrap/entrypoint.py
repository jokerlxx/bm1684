from __future__ import annotations

from app.application.orchestrator import SystemController, install_signal_handlers
from app.bootstrap.config import load_config
from app.interfaces.http.app_factory import create_app


def build_runtime(config_path="config_bm1684x.json"):
    controller = SystemController(config_path=config_path)
    install_signal_handlers(controller)
    app = create_app(
        controller.handlers(),
        get_scheduler=controller.get_scheduler,
        get_alert_queue=controller.get_alert_queue,
    )
    return controller, app


def main(config_path="config_bm1684x.json"):
    controller, app = build_runtime(config_path=config_path)
    config = load_config(config_path)
    port = config["output"].get("display_port", 5000)
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
    controller.shutdown()
