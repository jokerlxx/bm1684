# config_bm1684x.json 批调度与四路常驻调参

本文专门说明 `config_bm1684x.json` 中与 **批调度（`batch_scheduler`）**、**检测抽帧**、**预览流畅度** 相关的参数。这些参数决定「每路多久推理一次、四任务能否同时跑满」，**与算法阈值无关**。

算法误报/漏报、观察窗口、冷却等请见 [`config_bm1684x_algorithm_params.md`](config_bm1684x_algorithm_params.md)。

---

## 1. 概念：tick 与 Job

- **tick**：`BatchModelScheduler` 主循环的一轮。每轮从 FrameHub 取最新帧、挑选到期 **Job**、在预算内跑模型推理。
- **Job**：一个 `(通道, 检测器)` 单元，例如 `(ch3, ventilator)`。任务页勾选检测器并绑定通道后，对应 Job 进入调度表。
- **每 tick 推理预算**：由 `max_model_inferences_per_tick`（与 legacy 字段 `max_models_per_tick` 保持一致）决定。一轮里最多发起多少次 model 推理。

四路各绑一种检测器且同时运行时，**稳态下一 tick 往往需要 5 次推理**：

| 顺序 | model_key | 服务谁 |
| --- | --- | --- |
| 1 | `fight_detection` | ch1 fight |
| 2 | `fall_detection` | ch2 fall |
| 3 | `helmet_detection` | ch4 crowd |
| 4 | `ventilator_equipment` | ch3 ventilator（第一步） |
| 5 | `helmet_detection` | ch3 ventilator（第二步） |

若 `max_model_inferences_per_tick` 仅为 `2`，fight + fall 会占满预算，**ventilator / crowd 长期得不到推理**（日志里 `jobs=` 只剩 ch1、ch2）。现场已验证：改为 `5` 后四路可均衡到约 **3.4 次/秒/Job**。

---

## 2. 当前推荐基线（四路常驻）

适用于：**ch1 fight + ch2 fall + ch3 ventilator + ch4 crowd** 四任务同时开，fight 仍用 9b 模型。

```json
"batch_scheduler": {
  "enabled": true,
  "max_jobs_per_tick": 4,
  "max_model_inferences_per_tick": 5,
  "max_models_per_tick": 5,
  "max_batch_wait_ms": 0,
  "job_intervals_s": {
    "fall": 0.15,
    "fight": 0.15,
    "crowd": 0.2,
    "ventilator": 0.25,
    "helmet": 0.15,
    "window_door_inside": 2.5,
    "window_door_outside": 2.5
  },
  "detector_priority": {
    "ventilator": 10,
    "crowd": 10,
    "fall": 9,
    "fight": 9,
    "helmet": 9,
    "window_door_inside": 3,
    "window_door_outside": 3
  },
  "model_intervals_s": {
    "fall_detection": 0.15,
    "fight_detection": 0.15,
    "helmet_detection": 0.25,
    "ventilator_equipment": 0.25,
    "window_door_inside": 2.5,
    "window_door_outside": 2.5
  },
  "stream_interval_s": 0.12,
  "batch_streams": 1,
  "preview_backpressure_enabled": false
}
```

修改 `batch_scheduler` 后需**重启服务**或重新加载配置，并重新启停相关任务。

---

## 3. 参数速查

| 参数 | 当前推荐 | 作用 | 调大 / 调小 |
| --- | --- | --- | --- |
| `max_model_inferences_per_tick` | `5` | 每 tick 最多几次 model 推理。 | **四任务异构时低于 5 易饿死 ventilator/crowd**；再大增加单 tick 耗时与 TPU 占用。 |
| `max_models_per_tick` | `5` | 与上一项 legacy 对齐，请保持一致。 | 同上。 |
| `max_jobs_per_tick` | `4` | 每 tick 最多选几个 due Job。 | 四路常驻保持 `4`；单路可不改。 |
| `job_intervals_s.<detector>` | 见第 2 节 | 该检测器 Job 两次服务的最小间隔（秒）。 | **数值越小越勤**；fight/fall 过小会加重 TPU，过大则框更新慢。 |
| `model_intervals_s.<model_key>` | 与 job 间隔协调 | 模型维度的最小间隔。 | 建议与对应 `job_intervals_s` 同量级，避免两套间隔打架。 |
| `detector_priority` | ventilator/crowd=10 | due Job 相同时的优先顺序。 | 数值越大越优先；预算紧张时保护安全类。 |
| `window_door_*` 间隔 | `2.5` | 门窗为固定慢变场景。 | **应加长间隔（2–3s）省 TPU**；不宜短于 1s。 |
| `stream_interval_s` | `0.12` | 写入 timeshare 的通道最小更新间隔。 | 一般保持默认。 |
| `max_batch_wait_ms` | `0` | 为合批多等一会儿。 | 四路各绑不同 model 时保持 `0`。 |
| `preview_backpressure_enabled` | `false` | 预览 FPS 低时暂停推理。 | 默认关；打开后 TPU 可能长时间空闲。 |

---

## 4. `job_intervals_s` 分检测器说明

| 检测器 | 推荐间隔 | 说明 |
| --- | ---: | --- |
| `fight` / `fall` | `0.15` | 安全类；fight 单次推理 ~200ms，是 TPU 主负载。 |
| `crowd` | `0.20` | 复用 `helmet_detection`；与 `detect_emit_fps=3` 匹配。 |
| `ventilator` | `0.25` | 每个 Job 占 **2 次**推理（equipment + helmet），不宜再缩短。 |
| `helmet` | `0.15` | 独立安全帽任务时使用。 |
| `window_door_inside` / `outside` | `2.5` | 固定物体，少跑即可。 |

理论上限还受 `bm1684x.detect_emit_fps`（当前 3）限制：每路进 FrameHub 约 3fps，Job 间隔再小也难超过约 3 次有效更新/秒。

批调度下 `window_door_*` 间隔为 2.5s 时，`observation_frames=15` 约需 37s 墙钟时间才满观察窗口；若需更快报警可略减 `observation_frames`（见算法文档门窗节），或缩短间隔但会多占 TPU。

---

## 5. 日志验收（改完必看）

在 `bm-main.log` 中搜索 `Batch scheduler metrics`，四任务全开且运行 2–3 分钟后应接近：

```text
jobs=[ch1:fight≈3.2~3.4, ch2:fall≈3.2~3.4, ch3:ventilator≈3.2~3.4, ch4:crowd≈3.2~3.4]
infer=[fall_detection; fight_detection; helmet_detection; ventilator_equipment]
waste=0  skips(no_streams=0,no_frames=0,backpressure=0)
```

注意：

- `helmet_detection` 的 `n=` 约为 fight/fall 的 **2 倍**（crowd 一条 + ventilator 一条 helmet），属正常现象。
- `batch_avg=1.0` 表示每 model 每路单独推理；仅「多路同 model」时才会出现 `batch_avg=2`（如两路 fight 合批 9b）。

---

## 6. 常见现象与调参方向

| 现象 | 可能原因 | 建议 |
| --- | --- | --- |
| 某路长期无框、无报警，但任务已开 | `max_model_inferences_per_tick` 过小 | 提到 **≥5**（四任务异构基线） |
| `jobs=` 只有 fight/fall | 同上 | 同上 |
| 预览 FPS 明显偏低（长期 <10） | 四路推理 + 多路报警框 + 录像编码 | 略增 fight/fall 间隔至 `0.2`；或 `detect_emit_fps` 降至 `2` |
| 单 tick 延迟感强 | fight preprocess ~180ms × 多模型 | 预期行为；换 fight 1b 模型是模型侧优化，非本表参数 |
| 门窗任务占 TPU | 间隔过短 | 保持 `2.5~3.0`，`detector_priority` 保持 `3` |
| 测试 mp4 循环时预览卡一下 | 四路同时 EOF 重开解码器 | RTSP 长流无此问题；与调度无关 |

---

## 7. 保守档（预览/录像压力更大时）

在基线基础上可试：

| 参数 | 保守值 |
| --- | ---: |
| `max_model_inferences_per_tick` / `max_models_per_tick` | `6` |
| `job_intervals_s.fight` / `fall` | `0.20` |
| `job_intervals_s.crowd` | `0.25` |
| `job_intervals_s.ventilator` | `0.30` |
| `bm1684x.detect_emit_fps` | `2` |

---

## 8. 采集、预览与其它影响项

这些不是算法阈值，但会影响检测帧率、画面流畅度与坐标尺度：

| 配置项 | 当前值 | 影响 |
| --- | ---: | --- |
| `fps` | `20` | 系统输入帧率参考值。 |
| `bm1684x.detect_emit_fps` | `3` | 每路写入 FrameHub 的检测抽帧上限；四路常驻建议 2–3。 |
| `bm1684x.detect_frame_max_width` | `640` | 检测侧最大宽度，影响 `eps_pixels` 等像素阈值（算法文档）。 |
| `bm1684x.detect_frame_max_height` | `360` | 检测侧最大高度。 |
| `bm1684x.scale_backend` | `cv2` | 预览缩放后端；与推理 BMCV 隔离，减轻资源争用。 |
| `batch_scheduler.batch_streams` | `1` | 合批/stream 相关 legacy；四路 Job 模式下保持默认。 |
| `detection_timeshare.tick_sleep_s` | `0.005` | 调度空转 sleep，一般不改。 |
| `output.preview_alert_box_hold_s` | `1.2` | 预览报警框短暂丢失时的保持时间；**不改变是否报警**。 |
| `output.preview_fps` | `20` | 预览输出目标 FPS。 |
| `output.alarm_buffer_fps` | 见 JSON | 告警录像缓冲帧率上限，不影响网页预览。 |

---

## 9. 与算法参数的分工

| 问题类型 | 改哪里 |
| --- | --- |
| 误报 / 漏报、观察窗口、冷却 | [`config_bm1684x_algorithm_params.md`](config_bm1684x_algorithm_params.md) 各 `*_detection` 节 |
| 某路完全不做推理、四路不均衡 | 本文 `batch_scheduler` |
| 有推理但预览框闪 | 本文第 8 节 `output.preview_alert_box_hold_s` |
| 像素距离类阈值不准 | 本文第 8 节 `bm1684x.detect_frame_max_*` |

---

## 10. 调参检查清单（运维向）

1. 确认四路任务已绑定正确通道，检测器已勾选。
2. 核对 `max_model_inferences_per_tick` / `max_models_per_tick` **≥ 5**（四任务异构）。
3. 改完配置后重启服务，重新启停任务。
4. 运行 2–3 分钟，按第 5 节检查 `Batch scheduler metrics`。
5. 若仍有某路无框，再查算法阈值（算法文档），不要先动阈值。
