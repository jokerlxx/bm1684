#!/usr/bin/env bash
set -euo pipefail

SERVICE_NAME="bm-web.service"
DST_SERVICE="/etc/systemd/system/${SERVICE_NAME}"

if [[ "${EUID:-$(id -u)}" -ne 0 ]]; then
  echo "请使用 root 运行，例如：sudo $0" >&2
  exit 1
fi

echo "停止并禁用 ${SERVICE_NAME}"
systemctl disable --now "${SERVICE_NAME}" || true

if [[ -f "${DST_SERVICE}" ]]; then
  echo "删除 ${DST_SERVICE}"
  rm -f "${DST_SERVICE}"
fi

echo "systemd 重新加载"
systemctl daemon-reload

echo "完成"
