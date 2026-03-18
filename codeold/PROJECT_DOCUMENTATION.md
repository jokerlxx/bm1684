# 多任务检测系统 - 工程说明文档

本文档面向开发与运维人员，说明工程整体结构、配置方式、**如何新增检测模型**以及**如何修改界面字体与样式**。

---

## 一、工程概述

### 1.1 功能简介

本系统为**多任务检测系统（时间段自动开关版）**，基于事件驱动的微服务架构，主要能力包括：

- **多路视频接入**：支持 1 / 2 / 4 / 9 / 16 路 RTSP 或本地视频文件，宫格展示。
- **六类检测**：跌倒、安全帽、打架、聚集、窗户门、呼吸机（均可独立启停与时间段调度）。
- **时间段自动调度**：为每个检测器配置开始/结束时间，支持跨天；到点自动开启/关闭，也可手动覆盖。
- **告警与存储**：告警前后视频与关键帧图片保存至 `alarm_videos`，支持 Web 端按类别查看与下载。
- **Web 控制台**：登录（admin/admin）、主画面实时预览、报警事件列表、模型报警参数与视频流配置。

### 1.2 技术栈与运行环境

- **Python**：3.8+
- **硬件/推理**：BM1684X + Sophon SAIL，bmodel 格式 YOLO 模型
- **Web**：Flask（MJPEG 视频流 + SSE 告警推送），前端单页 `frontend/index.html`（HTML/CSS/JS）
- **时区**：业务时间使用北京时区（Asia/Shanghai）

---

## 二、目录结构

```
项目根目录/
├── main.py                    # 主入口：MainScheduler、TimeslotScheduler、API 实现与 Flask 启动
├── config_bm1684x.json        # 主配置：视频源、路数、模型路径、各检测器报警参数、队列与输出
├── timeslot_config.json       # 时间段配置：各检测器启用与否及开始/结束时间
├── ARCHITECTURE.md            # 架构说明（事件与队列、模块对应关系）
├── PROJECT_DOCUMENTATION.md   # 本文档
│
├── backend/
│   ├── __init__.py
│   └── app.py                 # Flask 应用：/video_feed、/api/*、SSE /api/events、静态前端
│
├── core/
│   ├── __init__.py
│   ├── events.py              # 事件数据结构（若抽离）
│   ├── bm1684x_yolo_adapter.py # BM1684X YOLO 推理适配（bmodel 加载与推理）
│   └── display_service.py    # 展示服务：多路合成、告警规则、录制、alert_queue 推送
│
├── detection/
│   ├── __init__.py            # MODEL_POOL 与 run_xxx_detector 导出
│   ├── fall_detector.py
│   ├── helmet_detector.py
│   ├── fight_detector.py
│   ├── crowd_detector.py
│   ├── window_door_detector.py
│   └── ventilator_detector.py
│
├── ingestion/
│   ├── __init__.py
│   └── stream_service.py      # 视频流接入：读 RTSP/本地视频，写入 frame_queue
│
├── storage/
│   └── __init__.py            # 告警目录解析、list_alarm_files 等（供 backend 使用）
│
├── frontend/
│   └── index.html             # 单页前端：登录、预览/报警事件/配置、控制与样式
│
├── alarm_videos/              # 告警视频与图片输出目录（可配置）
├── simhei.ttf                 # 中文标注字体（可配置路径）
└── font.txt                   # 字体说明等（可选）
```

---

## 三、配置文件说明

### 3.1 config_bm1684x.json

| 配置项 | 说明 |
|--------|------|
| `input_mode` | 0=RTSP，1=视频文件 |
| `stream_count` | 1 / 2 / 4 / 9 / 16 路 |
| `rtsp_url` / `rtsp_urls` | 一路用前者，多路用数组 |
| `input_video_path` / `input_video_paths` | 文件模式下的路径 |
| `video_loop` | 文件是否循环播放 |
| `fps` | 目标帧率 |
| `models` | 各模型 bmodel 路径（见下文“新增模型”） |
| `fall_detection`、`helmet_detection` 等 | 各检测器报警参数（置信度、冷却时间等），Web 配置页会读写 |
| `output.video_output_dir` | 告警视频/图片目录 |
| `output.display_port` | Web 服务端口 |
| `output.font_path` | 画面中文标注字体路径 |
| `queue_sizes` | frame_queue / result_queue / display_queue 容量 |
| `bm1684x` | device_id、enable_sophon |

### 3.2 timeslot_config.json

每个检测器一项，键为检测器名（如 `fall`、`helmet`）：

- `enabled`：是否启用时间段调度  
- `start` / `end`：开始、结束时间（如 `"08:00"`、`"18:00"`），支持跨天。

---

## 四、如何新增检测模型（完整步骤）

新增一种检测类型（例如 `smoke` 烟雾检测）需要按顺序完成以下步骤，保证主控、检测进程、展示服务、配置与前端一致。

### 4.1 约定命名

- **检测器标识**：小写英文，如 `smoke`（与队列键、API、前端一致）。
- **模型配置键**：如 `models.smoke_detection`。
- **参数配置键**：如 `smoke_detection`（与 `config_bm1684x.json` 内段落一致）。

### 4.2 步骤 1：配置文件

**1）config_bm1684x.json**

- 在 `models` 中增加模型路径，例如：
  ```json
  "models": {
    "smoke_detection": "/path/to/smoke_fp32_1b.bmodel"
  }
  ```
- 在文件内增加该检测器的报警参数段落（供 Web 配置页读取/保存），例如：
  ```json
  "smoke_detection": {
    "conf_threshold": 0.3,
    "cooldown_duration": 180
  }
  ```

**2）timeslot_config.json**

- 在默认配置中增加一项（与 `main.py` 中 `TimeslotConfig.load` 的 `default_config` 一致）：
  ```json
  "smoke": { "enabled": false, "start": "08:00", "end": "18:00" }
  ```

### 4.3 步骤 2：检测模块 detection/

**1）新建 detection/smoke_detector.py**

- 实现入口函数：`run_smoke_detector(model_path, frame_queue, result_queue, control_queue, config)`。
- 从 `frame_queue` 取帧（与现有 detector 相同数据结构），用 `core.bm1684x_yolo_adapter` 或现有适配器做推理。
- 将结果放入 `result_queue`，格式需与 `core/display_service.py` 中使用的结构一致（如含 `detector_type`、`detections`、`frame`、`enabled` 等）。
- 监听 `control_queue`，收到 `stop` 退出，收到 `enable` 等做启用/禁用逻辑（若需要）。
- 从 `config` 中读取 `config['smoke_detection']` 作为报警参数（置信度、冷却时间等）。

可参考 `detection/helmet_detector.py` 或 `fall_detector.py` 的循环结构、队列收发和结果格式。

**2）detection/__init__.py**

- 增加导入与导出：
  ```python
  from detection.smoke_detector import run_smoke_detector
  MODEL_POOL["smoke"] = run_smoke_detector
  __all__ = [..., "run_smoke_detector"]
  ```

### 4.4 步骤 3：主控 main.py

在 `MainScheduler` 中统一增加“队列、进程、状态”三处，并在 `start_detector` / `stop_detector` 中增加分支。

**1）队列与状态（__init__）**

- `self.result_queues['smoke'] = mp.Queue(...)`
- `self.control_queues['smoke'] = mp.Queue()`
- `self.detector_running['smoke'] = False`

**2）start_detector**

- 在 `elif detector_name == 'window_door':` 之后增加：
  ```python
  elif detector_name == 'smoke':
      process = mp.Process(
          target=run_smoke_detector,
          args=(
              self.config['models']['smoke_detection'],
              self.frame_queue,
              self.result_queues['smoke'],
              self.control_queues['smoke'],
              self.config,
          ),
          daemon=True,
      )
  ```
- 若检测器需要多个模型或不同参数，按现有 `ventilator` 等方式调整 `args`。

**3）stop_detector**

- 无需改逻辑，只要 `detector_name` 在 `self.processes` 与 `control_queues` 中存在即可。

**4）时间段调度**

- 在 `TimeslotScheduler._schedule_loop` 的检测器列表中加入 `'smoke'`。
- 在 `TimeslotConfig.load` 的 `default_config` 中加入 `'smoke': {...}`（若尚未通过 timeslot_config.json 持久化）。

**5）API 与配置**

- `get_status_impl`：在遍历的检测器列表中加上 `'smoke'`。
- `toggle_detector_impl`：允许的检测器列表中加入 `'smoke'`。
- `save_timeslot_impl` / `get_all_timeslots_impl` / `check_timeslot_impl`：允许的检测器列表中加入 `'smoke'`。
- `get_config_alarm_params_impl`：在 `keys` 列表中加入 `'smoke_detection'`。
- `save_config_alarm_params_impl`：在 `keys` 列表中加入 `'smoke_detection'`。

### 4.5 步骤 4：展示服务 core/display_service.py

- 在“按检测器类型汇总结果并渲染”的逻辑中，增加对 `detector_type == 'smoke'` 的分支（若有特殊绘制逻辑）。
- 在告警判定与录制逻辑中，增加对 `smoke` 的规则（例如某类别置信度超过阈值且持续 N 秒则触发告警），并调用现有的录制与 `alert_queue.put(AlertEvent(...))`。
- 确保写入的告警事件中 `alarm_type` 或等价字段为 `'smoke'`，以便前端与存储按类别筛选。

### 4.6 步骤 5：前端 frontend/index.html

**1）主画面 - 检测器卡片**

- 在“日常场景”或“工作场景”的 `detector-buttons` 中增加一张卡片：
  - `id="smoke-toggle"`、`id="smoke-timeslot-enabled"`、`id="smoke-start"`、`id="smoke-end"` 等，与现有检测器一致。
  - `onchange="toggleDetector('smoke', this.checked)"`、`saveTimeslot('smoke')`、`updateTimeslotStatus('smoke')` 等。

**2）报警事件页**

- `#alerts-category` 的 `<select>` 中增加：`<option value="smoke">烟雾检测</option>`。
- JS 中 `alarmTypeLabels` 增加：`'smoke': '烟雾检测'`。

**3）配置页 - 模型报警参数**

- 后端已通过 `get_config_alarm_params_impl` 返回 `smoke_detection`，前端 `paramLabelMap` 中增加各参数的“中文（英文）”映射，例如：
  ```javascript
  conf_threshold: '置信度',
  cooldown_duration: '冷却时长(秒)',
  // ... 其他 smoke_detection 中使用的键
  ```
- 若配置页是按后端返回的 key 动态生成表单项的，则无需改 HTML，只需补全 `paramLabelMap`。

**4）状态与时间段**

- `getDetectorDisplayName`、`enableDetectorToggles`、`disableDetectorToggles`、`resetAllToggles`、`loadTimeslotConfigs`、`updateTimeslotStatusDisplay` 等使用的检测器数组中加入 `'smoke'`。

完成以上步骤后，新检测器即可：被主控启停、参与时间段调度、在配置页调节参数、在报警事件页按类别筛选与下载，并在主画面显示状态与时间端。

---

## 五、如何修改界面字体与样式

前端为单页 `frontend/index.html`，样式集中在文件内 `<style>` 中，通过 **CSS 变量** 和 **类名** 控制字体、字号和配色。

### 5.1 全局字体与变量（:root）

在 `<style>` 开头的 `:root` 中可统一调整：

| 变量名 | 含义 | 示例值 |
|--------|------|--------|
| `--font-sans` | 全局无衬线字体族 | `'Noto Sans SC', 'PingFang SC', 'Microsoft YaHei', sans-serif` |
| `--color-text` | 主文字颜色 | `#1e293b` |
| `--color-text-muted` | 次要/说明文字颜色 | `#64748b` |
| `--color-primary` | 主色（按钮、链接、强调） | `#4f46e5` |
| `--color-success` | 成功/启动 | `#059669` |
| `--color-danger` | 危险/停止/告警 | `#dc2626` |

修改字体族时，只需改 `--font-sans`；若引入其他 Google 字体，在 `<head>` 中增加对应 `<link>`，再把字体名写入 `--font-sans` 即可。

### 5.2 各部分字体大小速查

以下为当前使用的主要字号与位置，按“区域 + 选择器”列出，便于定向放大/缩小。

| 区域 | 选择器或位置 | 当前字号 | 说明 |
|------|----------------|----------|------|
| 侧栏导航 | `.nav-item` | 12px | 导航项文字 |
| 主标题 | `.container h1`、`h1` | 26px | 主画面标题 |
| 副标题 | `.subtitle` | 14px | 主画面副标题 |
| 场景标题 | `.scene-title` | 18px | “日常场景”“工作场景” |
| 场景图标 | `.scene-icon` | 22px | 场景前图标 |
| 检测器卡片标题 | `.detector-card h3` | 16px | 如“跌倒检测”“安全帽检测” |
| 检测器描述 | `.detector-description` | 14px | 卡片内说明文字 |
| 时间段标签 | `.timeslot-label` | 14px | “自动调度”等 |
| 时间输入/分隔 | `.time-input`、`.time-separator` | 14px | 时间输入框与“至” |
| 时间段状态 | `.timeslot-status` | 13px | “时间段未启用”等 |
| 主按钮 | `.btn` | 14px | 启动/停止/刷新等 |
| 保存时间段 | `.btn-save-timeslot` | 13px | “保存”按钮 |
| Toast 告警 | `.toast`、`.toast .toast-time` | 14px / 12px | 右上角弹窗 |
| 报警事件工具栏 | `.alerts-toolbar label` | 14px | “告警类别”等 |
| 报警事件路径提示 | `.alerts-path` | 13px | 保存路径说明 |
| 告警卡片标题 | `.alerts-list .alert-card .name` | 13px | 文件名 |
| 告警卡片元信息 | `.alerts-list .alert-card .meta` | 12px | 类别、时间 |
| 告警卡片按钮 | `.alerts-list .alert-card .btn-download` | 13px | “下载到本地” |
| 配置标题 | `.config-toolbar h2` | 18px | “相关配置” |
| 配置区块标题 | `.config-section-title` | 15px | 如“跌倒检测”区块 |
| 配置项标签 | `.config-field label` | 13px | 如“置信度（conf_threshold）” |
| 配置项输入框 | `.config-field input` | 14px | 输入框文字 |
| 保存配置按钮 | `.btn-save-config` | 14px | “保存视频流配置”等 |
| 空状态说明 | `.empty-state`、`.empty-state-icon` | 15px / 48px | 无告警时的提示与图标 |
| 登录页 Logo | `.login-logo` | 24px | “多任务检测系统” |
| 登录页副标题 | `.login-sub` | 14px | “实时监测 · ...” |
| 登录页主标题 | `.login-title` | 26px | “欢迎登录” |
| 登录页输入框 | `.login-form .field-wrap input` | 15px | 用户名/密码 |
| 登录页按钮 | `.login-btn` | 16px | “登录” |
| 登录页免责/版权 | `.login-disclaimer`、`.login-copyright` | 12px | 底部小字 |

如需“整体放大一级”：可先增加一个全局基础字号变量（如 `--font-size-base: 15px`），再在需要处使用 `calc(var(--font-size-base) * 1.1)` 等；或直接在上表对应选择器中把 `font-size` 改大 1～2px。

### 5.3 修改方式建议

1. **只改某一类文字**：在上表中找到对应选择器，在 `<style>` 中改该选择器的 `font-size`（及可选的 `font-weight`、`line-height`）。
2. **整站统一变大/变小**：在 `:root` 中增加 `--font-size-base: 14px`（或 16px），然后在各区域用 `font-size: var(--font-size-base)` 或 `calc(var(--font-size-base) + 2px)` 等替代固定 px。
3. **仅改主画面控制区**：集中改 `.control-section` 下 `.scene-title`、`.detector-card h3`、`.detector-description`、`.timeslot-label`、`.timeslot-status`、`.btn` 等的 `font-size`。
4. **仅改配置页**：改 `.config-toolbar h2`、`.config-section-title`、`.config-field label`、`.config-field input`、`.btn-save-config` 的 `font-size`。

修改后保存 `frontend/index.html` 并刷新浏览器即可生效（无需重启后端）。

---

## 六、运行与部署

### 6.1 启动

```bash
# 在项目根目录
python main.py
```

- 默认 Web：`http://0.0.0.0:5000`（端口由 `config_bm1684x.json` 的 `output.display_port` 决定）。
- 默认账号：用户名 `admin`，密码 `admin`。

### 6.2 流程简述

1. 启动后仅 Web 与时间段调度就绪，视频流与检测器未启动。
2. 用户在 Web 点击“启动系统”：主控拉起流服务与展示服务，时间段调度开始按配置检查。
3. 用户可为每个检测器勾选“启用”并配置时间段；到时间会自动开启/关闭，也可随时手动开关。
4. 告警触发后，展示服务保存视频与图片到 `alarm_videos`，并通过 SSE 推送到前端；用户在“报警事件”页按类别筛选并下载。

### 6.3 日志与排错

- 日志输出到标准输出，格式：`[时间] [模块名] [级别] 消息`。
- 若某检测器无法启动，请检查：`config_bm1684x.json` 中对应模型路径是否正确、BM1684X 驱动与 Sophon 是否可用、该检测器进程的日志是否有报错。

---

## 七、附录：API 一览（供联调与扩展）

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/` | 前端静态页 |
| GET | `/video_feed` | MJPEG 视频流 |
| GET | `/api/events` | SSE 告警事件流 |
| POST | `/api/start` | 启动系统 |
| POST | `/api/stop` | 停止系统 |
| POST | `/api/toggle_detector` | 启/停单个检测器（body: detector, enabled） |
| GET | `/api/status` | 系统及检测器运行状态 |
| POST | `/api/timeslot/save` | 保存时间段配置 |
| GET | `/api/timeslot/get_all` | 获取全部时间段配置 |
| GET | `/api/timeslot/check` | 查询某检测器是否在时间段内 |
| GET | `/api/alerts/history` | 告警文件列表（含 category 筛选） |
| GET | `/api/alerts/file?name=xxx` | 下载告警文件 |
| GET | `/api/config/stream` | 获取视频流配置 |
| POST | `/api/config/stream` | 保存视频流配置 |
| GET | `/api/config/alarm_params` | 获取各模型报警参数 |
| POST | `/api/config/alarm_params` | 保存各模型报警参数 |

---

文档版本与工程保持一致，若后续增删检测器或调整前端结构，请同步更新本文档与 ARCHITECTURE.md。
