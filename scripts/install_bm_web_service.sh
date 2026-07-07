#!/usr/bin/env bash
set -euo pipefail

SERVICE_NAME="bm-web.service"
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_SERVICE="${SRC_DIR}/deploy/${SERVICE_NAME}"
DST_SERVICE="/etc/systemd/system/${SERVICE_NAME}"

if [[ ! -f "${SRC_SERVICE}" ]]; then
  echo "ERROR: service file not found: ${SRC_SERVICE}" >&2
  exit 1
fi

if [[ "${EUID:-$(id -u)}" -ne 0 ]]; then
  echo "请使用 root 运行，例如：sudo $0" >&2
  exit 1
fi

echo "安装 ${SERVICE_NAME} -> ${DST_SERVICE}"
cp -f "${SRC_SERVICE}" "${DST_SERVICE}"

echo "systemd 重新加载"
systemctl daemon-reload

echo "设置开机自启并立即启动"
systemctl enable --now "${SERVICE_NAME}"

echo
echo "完成。常用命令："
echo "  查看状态: systemctl status ${SERVICE_NAME} --no-pager"
echo "  查看日志: journalctl -u ${SERVICE_NAME} -f"
echo "  重启服务: systemctl restart ${SERVICE_NAME}"
