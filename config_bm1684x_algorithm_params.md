# config_bm1684x.json 参数说明

本文说明 `config_bm1684x.json` 的调参方式，分为两部分：

1. **算法参数**（第 1–6 节）：各检测器的阈值、观察窗口、跟踪与冷却等，决定「报不报警、何时报警」。
2. **批调度与四路常驻**：见独立文档 [`config_bm1684x_batch_scheduler.md`](config_bm1684x_batch_scheduler.md)（`batch_scheduler`、`detect_emit_fps`、预览相关项）。

当前系统默认是四路预览模式：预览页固定展示 2x2 四宫格，任务可绑定通道 1–4 中的一路，并可同时启用 `fall`、`fight`、`crowd`、`helmet`、`ventilator`、`window_door_inside` 和 `window_door_outside`；现场可在任务页勾选需要运行的算法。

`ventilator` 在界面和任务配置中仍算 1 个业务算法，但内部会同时使用呼吸器设备模型和头部模型。

## 1. 推荐调参方式

日常只建议通过 `config_bm1684x.json` 调整各算法主配置节里的参数。任务管理页面只负责选择视频通道和检测器，不再提供在线调参入口。

代码仍兼容旧配置里的 `advanced_algorithm_params` 覆盖区，但默认配置不再使用它；如果现场临时加了同名高级参数，高级区的值会覆盖主配置节。

## 2. 算法对应关系

| 业务算法 | `tasks[].detectors` | 基础配置节 | 模型 key | 说明 |
| --- | --- | --- | --- | --- |
| 跌倒检测 | `fall` | `fall_detection` | `fall_detection` | 检测跌倒目标，并按时间窗口确认报警。 |
| 打架检测 | `fight` | `fight_detection` | `fight_detection` | 检测打架目标，并按全局事件确认报警。 |
| 聚集检测 | `crowd` | `crowd_detection` | `helmet_detection` | 复用头部/人员检测结果做聚类。 |
| 安全帽检测 | `helmet` | `helmet_detection` | `helmet_detection` | class_id=1 视为未戴安全帽。 |
| 呼吸器检测 | `ventilator` | `ventilator_detection` | `ventilator_equipment` + `helmet_detection` | 设备模型找面罩/氧气瓶，头部模型定位人员。 |
| 仓内门窗 | `window_door_inside` | `window_door_detection` | `window_door_inside` | 门开、窗开报警。 |
| 仓外门窗 | `window_door_outside` | `window_door_detection` | `window_door_outside` | 与仓内门窗共用同一套参数。 |

## 3. 基础参数速查

| 参数 | 常见单位/范围 | 作用 | 调参方向 |
| --- | --- | --- | --- |
| `conf_threshold` | 0-1 | 模型基础置信度阈值。 | 误检多调高，漏检多调低。 |
| `equipment_conf` | 0-1 | 呼吸器设备模型的面罩/氧气瓶置信度阈值。 | 设备误检多调高，设备漏检多调低。 |
| `person_conf` | 0-1 | 聚集检测中参与聚类的人员/头部置信度阈值。 | 杂框多调高，漏人多调低。 |
| `observation_duration` | 秒 | 最近多少秒作为事件统计窗口。 | 调大更稳但更慢，调小更快但更容易抖。 |
| `min_observation_duration` / `alert_duration` | 秒 | 至少观察多久才允许报警。 | 瞬时误报多调大，需要更快报警调小。 |
| `*_threshold` / `stability_ratio` | 0-1 | 观察窗口内命中比例达到多少才报警。 | 误报多调高，漏报多调低。 |
| `observation_frames` | 帧 | 门窗检测的观察帧数。 | 报警慢调小，误报多调大。 |
| `track_max_age` | 帧 | 跟踪目标在短暂丢检后保留多久。 | 框容易断调大，错跟/粘连调小。 |
| `track_iou_threshold` / `fall_match_iou_threshold` | 0-1 | 跟踪或跌倒框匹配所需的重叠比例。 | 目标切换错配调高，框抖导致断跟调低。 |
| `eps_pixels` | 像素 | 聚集检测中人员中心点的聚类距离。 | 聚不起来调大，远距离误聚调小。 |
| `min_samples` | 人数 | 聚集检测至少多少人形成聚集。 | 想更少人数报警调小，只关心多人聚集调大。 |
| `cooldown_duration` | 秒 | 同一事件报警后的冷却时间。 | 重复报警多调大，需要更频繁记录调小。 |

> 比例参数按 0-1 写，例如 `0.7` 表示 70%。像素参数按检测坐标系理解，当前检测帧会受 `bm1684x.detect_frame_max_width` 和 `detect_frame_max_height` 影响。

## 4. 默认基础配置

### 跌倒检测：`fall_detection`

```json
"fall_detection": {
  "conf_threshold": 0.25,
  "observation_duration": 1.5,
  "min_observation_duration": 1.0,
  "fall_threshold": 0.5,
  "large_bbox_conf": 0.3,
  "small_bbox_conf": 0.3,
  "bbox_area_split": 50000,
  "track_max_age": 30,
  "track_iou_threshold": 0.3,
  "fall_match_iou_threshold": 0.3,
  "cooldown_duration": 180
}
```

报警逻辑：最近 `observation_duration` 秒内，跌倒命中比例达到 `fall_threshold`，且观察跨度达到 `min_observation_duration` 后报警。

`conf_threshold` 是基础置信度阈值；`large_bbox_conf` / `small_bbox_conf` 会按目标框面积做更严格的二次过滤，面积分界由 `bbox_area_split` 控制。`track_max_age` 和 `track_iou_threshold` 控制目标跟踪稳定性，`fall_match_iou_threshold` 控制检测出的跌倒框与跟踪目标是否算同一个目标。

### 打架检测：`fight_detection`

```json
"fight_detection": {
  "conf_threshold": 0.5,
  "observation_duration": 2.0,
  "min_observation_duration": 1.0,
  "fight_threshold": 0.4,
  "cooldown_duration": 180
}
```

报警逻辑：最近 `observation_duration` 秒内出现打架目标的比例达到 `fight_threshold`，且满足最短观察时间后报警。

### 聚集检测：`crowd_detection`

```json
"crowd_detection": {
  "person_conf": 0.25,
  "eps_pixels": 70,
  "min_samples": 3,
  "observation_duration": 2.0,
  "stability_ratio": 0.6,
  "stability_duration": 1,
  "spatial_distance": 50,
  "cooldown_duration": 30
}
```

报警逻辑：先按 `eps_pixels` 和 `min_samples` 聚类，再按 `observation_duration` 和 `stability_ratio` 做稳定确认。

### 安全帽检测：`helmet_detection`

```json
"helmet_detection": {
  "conf_threshold": 0.35,
  "iou_threshold": 0.3,
  "max_age": 2,
  "center_distance_threshold_ratio": 1.6,
  "observation_duration": 1.5,
  "no_helmet_threshold": 0.7,
  "alert_duration": 1,
  "timer_reset_grace_s": 1.0,
  "track_timeout_s": 1.2,
  "cooldown_duration": 180
}
```

报警逻辑：同一目标最近 `observation_duration` 秒内未戴帽比例达到 `no_helmet_threshold`，且观察跨度达到 `alert_duration` 后报警。

### 呼吸器检测：`ventilator_detection`

```json
"ventilator_detection": {
  "equipment_conf": 0.3,
  "observation_duration": 2.0,
  "min_observation_duration": 1.5,
  "missing_equipment_threshold": 0.8,
  "pass_threshold": 0.2,
  "mask_iou_threshold": 0.15,
  "tank_distance_coefficient": 2,
  "tank_x_offset_coefficient": 1,
  "cooldown_duration": 180
}
```

报警逻辑：头部目标在最近 `observation_duration` 秒内缺少面罩/氧气瓶的比例达到 `missing_equipment_threshold`，且观察跨度达到 `min_observation_duration` 后报警。

匹配参数：

| 参数 | 说明 |
| --- | --- |
| `pass_threshold` | 佩戴率展示/判定指标；报警主要看 `missing_equipment_threshold`。 |
| `mask_iou_threshold` | 面罩框与头部框的匹配 IoU。 |
| `tank_distance_coefficient` | 氧气瓶中心到头部中心的最大距离系数。 |
| `tank_x_offset_coefficient` | 氧气瓶相对头部的横向偏移系数。 |

### 门窗检测：`window_door_detection`

```json
"window_door_detection": {
  "conf_threshold": 0.25,
  "iou_threshold": 0.3,
  "max_age": 2,
  "observation_frames": 15,
  "detection_threshold": 0.5,
  "cooldown_duration": 180
}
```

报警逻辑：只对窗户开、门开确认报警；同一目标在 `observation_frames` 个算法更新中，报警类出现比例达到 `detection_threshold` 后报警。

> 批调度下 `window_door_*` 的 `job_intervals_s` 为 2.5s 时，15 帧观察约需 37s 墙钟时间；若需更快报警可略减 `observation_frames`（如 10），或缩短间隔但会多占 TPU。详见 [`config_bm1684x_batch_scheduler.md`](config_bm1684x_batch_scheduler.md) 第 4 节。

门窗类别映射：

| class_id | 标签 | 显示名 | 是否报警 |
| ---: | --- | --- | --- |
| `0` | `window_open` | 窗户开 | 是 |
| `1` | `window_close` | 窗户关 | 否 |
| `2` | `door_close` | 门关 | 否 |
| `3` | `door_open` | 门开 | 是 |

## 5. 匹配/跟踪参数速查

| 参数 | 所属算法 | 说明 |
| --- | --- | --- |
| `large_bbox_conf` / `small_bbox_conf` | 跌倒 | 按目标框大小做二次置信度过滤。 |
| `stability_duration` | 聚集 | 未显式配置最短观察时间时的默认参考值。 |
| `spatial_distance` | 聚集 | 判断是否同一聚集位置的中心点距离阈值。 |
| `iou_threshold` | 安全帽/门窗 | 跟踪或 NMS 匹配阈值。 |
| `max_age` | 安全帽/门窗 | 目标允许连续丢失的最大更新次数。 |
| `center_distance_threshold_ratio` | 安全帽 | IoU 不足时按中心距离匹配目标的阈值。 |
| `timer_reset_grace_s` | 安全帽 | 未戴帽状态短暂消失时保留计时的宽限时间。 |
| `track_timeout_s` | 安全帽 | 目标多久未出现后清理状态。 |

旧配置中的 `ventilator_detection.person_conf`、`ventilator_detection.fail_threshold`、`ventilator_detection.use_helmet_detection`、`fight_detection.alarm_frames`、`helmet_detection.min_update_interval_s` 不再作为推荐调参项。当前默认配置已移除这些历史参数。

## 6. 调参检查清单

1. 先确认 `tasks[].stream_index` 已绑定目标通道，`tasks[].detectors` 已勾选当前要运行的算法。
2. **四路多任务同时开时**，先按 [`config_bm1684x_batch_scheduler.md`](config_bm1684x_batch_scheduler.md) 核对 `batch_scheduler`（尤其 `max_model_inferences_per_tick≥5`），再看算法阈值。
3. 误检/漏检优先调 `conf_threshold`、`equipment_conf` 或 `person_conf`。
4. 报警太快/太慢优先调 `observation_duration`、`min_observation_duration`、`alert_duration` 或 `observation_frames`。
5. 误报/漏报再调比例阈值：`fall_threshold`、`fight_threshold`、`stability_ratio`、`no_helmet_threshold`、`missing_equipment_threshold`、`detection_threshold`。
6. 重复报警调 `cooldown_duration`。
7. 目标框 ID 跳变、匹配不稳、聚集位置合并错误，再调对应算法配置块里的跟踪/匹配参数。
8. 改完调度参数后，用批调度文档第 5 节的日志指标验收四路 `jobs=` 是否均衡。

## 7. 批调度与资源（另见专文）

`batch_scheduler`、`detect_emit_fps`、预览反压等与 **四路能否同时跑满、每路多久推理一次** 相关，与算法阈值分开维护。

**完整说明、基线 JSON、日志验收与故障排查** → [`config_bm1684x_batch_scheduler.md`](config_bm1684x_batch_scheduler.md)

现场快速核对：

| 项 | 四路常驻推荐 |
| --- | --- |
| `max_model_inferences_per_tick` | `5` |
| `job_intervals_s` fight/fall/crowd/ventilator | `0.15` / `0.15` / `0.2` / `0.25` |
| `window_door_*` 间隔 | `2.5` |
| `bm1684x.detect_emit_fps` | `3`（压力大可降至 `2`） |

某路长期无推理时，**先查批调度文档**，再调算法阈值。
