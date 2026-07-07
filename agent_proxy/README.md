# GrainAI Agent Proxy

粮库AI监控平台 — 上位机DeepSeek代理服务。

BM1684边缘盒子不连外网，Proxy部署在上位机(Windows)，负责：
- 接收前端AI助手请求
- 转发BM1684获取真实系统数据
- 查询类问题 → 本地快速回答
- 分析/报告类 → DeepSeek大模型增强
- DeepSeek失败 → 自动回退本地回答

## 文件说明

```
agent_proxy/
  .env.example        # 配置文件模板
  .env                # 实际配置（不提交git）
  proxy_server.py     # 主程序
  start_proxy.bat     # 一键启动（前台窗口）
  install_proxy_service.bat   # 注册Windows后台服务（需管理员）
  uninstall_proxy_service.bat # 卸载Windows服务（需管理员）
  logs/               # 日志目录
    proxy.log         # 运行日志
  README.md           # 本文件
```

## 配置

编辑 `.env` 文件：

```ini
BM1684_BASE_URL=http://192.168.150.5:5010   # BM1684盒子地址
LLM_ENABLE=true                              # 启用DeepSeek
LLM_API_KEY=sk-你的Key                       # DeepSeek API Key
LLM_BASE_URL=https://api.deepseek.com        # API地址
LLM_MODEL=deepseek-chat                      # 模型
LLM_TIMEOUT=8                                # 超时秒数
PROXY_PORT=7000                              # 代理端口
```

## 启动方式

### 方式A：双击运行（临时）

直接双击 `start_proxy.bat`，出现窗口后最小化即可。

### 方式B：Windows后台服务（推荐，开机自启）

**安装：** 右键 `install_proxy_service.bat` → 以管理员身份运行

**管理：**
- `Win+R` → `services.msc` → 找到 `GrainAI Agent Proxy`
- 或用命令：`nssm start/stop/restart GrainAI-AgentProxy`

**卸载：** 右键 `uninstall_proxy_service.bat` → 以管理员身份运行

**服务特性：**
- 开机自动启动
- 崩溃后5秒自动重启
- 日志输出到 `logs/stdout.log` 和 `logs/stderr.log`

## 查看日志

```cmd
type logs\proxy.log
:: 或者实时查看
powershell -Command "Get-Content logs\proxy.log -Wait -Tail 50"
```

## 测试

```bash
# 健康检查
curl http://127.0.0.1:7000/proxy/health

# 查询类问题（本地快速回答）
curl -X POST http://127.0.0.1:7000/proxy/agent/chat -H "Content-Type: application/json" -d "{\"message\":\"今天有多少报警？\"}"

# 分析类问题（DeepSeek增强）
curl -X POST http://127.0.0.1:7000/proxy/agent/chat -H "Content-Type: application/json" -d "{\"message\":\"根据今天的报警给出处置建议\"}"
```

## 故障排查

| 问题 | 检查 |
|------|------|
| 前端一直"思考中" | 检查Proxy窗口是否在运行、端口7000是否被占用 |
| DeepSeek返回超时 | 调大 `.env` 中 `LLM_TIMEOUT`，或设 `LLM_ENABLE=false` 关闭 |
| 服务无法启动 | 检查是否以管理员身份运行、Python是否在PATH中 |
| BM1684连不上 | 检查 `.env` 中 `BM1684_BASE_URL` 是否正确 |

## 安全说明

- API Key 仅保存在 `.env` 文件中，不提交到代码仓库
- 发送给DeepSeek的数据经过脱敏处理
- DeepSeek无法直接访问摄像头、RTSP流、BM1684设备
- `.env` 和 `logs/` 目录已加入 `.gitignore`
