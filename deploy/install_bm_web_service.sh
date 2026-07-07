#!/usr/bin/env bash
set -euo pipefail

SERVICE_SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/bm-web.service"
SERVICE_DST="/etc/systemd/system/bm-web.service"

if [[ ! -f "$SERVICE_SRC" ]]; then
  echo "找不到服务文件: $SERVICE_SRC" >&2
  exit 1
fi

echo "安装 systemd 服务: bm-web.service"
sudo cp "$SERVICE_SRC" "$SERVICE_DST"
sudo systemctl daemon-reload
sudo systemctl enable bm-web.service
sudo systemctl restart bm-web.service

echo
echo "完成。查看状态："
echo "  systemctl status bm-web.service --no-pager"
echo "查看日志："
echo "  journalctl -u bm-web.service -f"

