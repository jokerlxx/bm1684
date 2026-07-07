## 目标

在设备上把本工程作为 **Web 服务常驻运行**，无需再进入盒子手动执行 `python3 main.py`。

完成后在同网段电脑/手机浏览器直接访问：

- `http://192.168.150.5:5001/`

其中端口来自 `config_bm1684x.json` 的 `output.display_port`（当前为 `5001`）。

## 已有实现说明（你现在这份工程已经具备）

- **Web 入口**：`main.py` 会启动 Flask，并监听 `0.0.0.0:<display_port>`，因此可通过设备 IP 访问。
- **前端页面**：`frontend/index.html` 由后端路由 `/` 返回。
- **后端 API**：见 `backend/app.py`，包含启动/停止系统、状态查询、任务管理、视频流（MJPEG）等；实时预览由 `/video_feed` 提供。
- **自启动服务文件**：`deploy/bm-web.service`（systemd）。

## 方式 A（推荐）：systemd 开机自启

把服务安装到系统里后，设备每次开机都会自动启动 Web 服务。

在设备上执行（需要 sudo 权限）：

```bash
sudo cp /home/admin/workspace/code/code/deploy/bm-web.service /etc/systemd/system/bm-web.service
sudo systemctl daemon-reload
sudo systemctl enable bm-web.service
sudo systemctl start bm-web.service
```

查看状态与日志：

```bash
systemctl status bm-web.service --no-pager
journalctl -u bm-web.service -f
```

停止/重启：

```bash
sudo systemctl stop bm-web.service
sudo systemctl restart bm-web.service
```

## 方式 B：手动运行（临时）

```bash
cd /home/admin/workspace/code/code
python3 main.py
```

## 常见问题

- **浏览器打不开**
  - 确认设备 IP 是否为 `192.168.150.5`（用 `ip a` 查看）
  - 确认端口是否为 `5001`（看 `config_bm1684x.json` 的 `output.display_port`）
  - 确认防火墙/安全策略未拦截 `5001/tcp`

- **端口想改**
  - 修改 `config_bm1684x.json`：
    - `output.display_port`: 改为目标端口
  - 然后重启服务：`sudo systemctl restart bm-web.service`
