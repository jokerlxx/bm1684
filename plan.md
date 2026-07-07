# TPU 利用率与前处理链路优化计划

## 背景

当前系统 9 路 RTSP 输入在线，算法进程正在运行。`bm-smi` 看到 TPU 利用率偏低，但日志显示不是模型没有运行，而是 TPU 计算时间在总耗时里占比很小。

## 执行结果

执行分支：`optimize/preprocess-pipeline`

已完成：

- 阶段 0 已完成：采集到 9 路全开基线，确认 `engine.process` 只有约 6-10ms，主要瓶颈在 YOLO26 preprocess。
- 阶段 1 已完成：复用 YOLO26 input tensor、预处理槽位、输入 numpy buffer，并补充 preprocess/inference/postprocess/total 分段日志。
- 阶段 2 已完成一个兼容快路径：不跨进程传裸 `BMImage`，改为 DecodeHub 写入检测共享内存前将检测帧限制到 `640x360`，同时为 batch=9 但 SAIL 缺少 `BMImageArray9D` 的环境增加 `BMImageArray8D + 1` 混合复用。
- 阶段 3 已部分验证：预览链路继续独立，检测优化不依赖预览 FPS；预览在报警叠加和录像较多时仍是单独瓶颈。
- 阶段 4 未继续加压调参：当前 CPU/显示侧已有压力，继续提高 `detect_emit_fps` 或 `max_models_per_tick` 会先增加 CPU 和预览压力，暂不作为本轮提交内容。

实测摘要：

- 阶段 1 后，batch=9 典型 preprocess 从约 100-130ms 降到约 88-98ms，total 从约 120-126ms 降到约 96-109ms。
- 阶段 2 初版 `960x540` 无收益，因为源流实际约 `960x544`，只触发微小缩放并引入额外开销；已改为 `640x360`。
- 阶段 2 最终版后，batch=9 典型 preprocess 约 70-80ms，部分窗口达到 66-76ms；典型 total 约 80-90ms，部分模型窗口低于 80ms。
- 计划中的激进目标 `preprocess < 60ms`、`total < 80ms` 尚未全模型稳定达成。剩余固定开销主要来自 `numpy/shared memory -> BMImage -> BMCV preprocess -> Tensor` 这段链路，以及 batch=9 无原生 `BMImageArray9D` 只能走混合 fallback。
- 预览 `compose_fps` 在无大量报警叠加时可短时约 14-15fps，在报警框和录像保存较多时回落到约 4-7fps，是后续独立优化项。

验证命令：

```bash
python3 -m py_compile core/bm1684x_yolo26_adapter.py ingestion/decode_hub.py app/bootstrap/config.py app/pipeline/model_scheduler.py tests/ingestion/test_decode_hub.py
python3 -m json.tool config_bm1684x.json >/dev/null
python3 -m pytest tests/core/test_bm1684x_yolo26_adapter.py tests/app/test_model_scheduler.py tests/ingestion/test_decode_hub.py -q
```

验证备注：

- `py_compile` 和 JSON 校验通过。
- 当前环境缺少 pytest，`python3 -m pytest ...` 失败信息为 `No module named pytest`，未能运行单元测试。
- 现场 9 路全开验证通过，检测结果持续输出；未观察到坐标系异常，结果 payload 继续携带 `coord_width/coord_height`。

已观察到的运行态数据：

- DecodeHub 拉流基本稳定在每路 20fps。
- 启用算法：fall、fight、helmet、crowd、ventilator。
- 实际启用模型：fall_detection、fight_detection、helmet_detection、ventilator_equipment。
- 模型采用 9 路 batch 推理，batch size 为 9。
- YOLO26 单次 batch 总耗时约 110-150ms。
- 其中真正 `engine.process` 推理时间约 6-8ms。
- 大头耗时在 preprocess，约 100-140ms。
- 当前预览合成约 4-7fps，低于目标 20fps，CPU 侧也有明显压力。

核心判断：

TPU 利用率低的主要原因不是 batch 不够，也不是模型没调度，而是 `DecodeHub -> numpy/shared memory -> BMImage/Tensor -> engine.process` 链路中数据搬运和前处理耗时过高。TPU 每次只短暂工作几毫秒，之后等待 CPU/BMCV 前处理和调度。

## 目标

第一目标是提高检测链路吞吐，让 TPU 被更连续地喂数据。

可量化目标：

- YOLO26 batch=9 的 preprocess 平均耗时从约 100-140ms 降到 60ms 以下。
- YOLO26 batch=9 的 total 平均耗时从约 120ms 降到 80ms 以下。
- 保持 9 路在线时模型结果正常输出，报警框坐标不漂移。
- 在不明显牺牲预览稳定性的前提下，提高 `bm-smi` 中 TPU 忙碌度。

辅助目标：

- 日志能区分 preprocess、inference、postprocess、scheduler wait、display cost。
- 优化过程可分阶段回滚。
- 每个阶段都有明确验证命令和通过标准。

## 非目标

本轮不做以下事情：

- 不更换模型结构或重新导出 bmodel。
- 不改变算法业务逻辑和报警阈值。
- 不重写整个进程架构。
- 不以提高 `bm-smi` 数字为唯一目标牺牲检测正确性。
- 不把预览优化和检测前处理优化混在一个大改动里。

## 当前关键链路

检测侧数据路径：

1. `ingestion/decode_hub.py` 使用 `sail.MultiDecoder` 解码 RTSP。
2. DecodeHub 将 `BMImage` 通过 `bmimg.asmat()` 转成 numpy。
3. `LatestFrameHub` 将 numpy frame 写入共享内存。
4. `app/pipeline/model_scheduler.py` 从共享内存读取 numpy frame，按模型和通道组 batch。
5. `core/bm1684x_yolo26_adapter.py` 将 numpy frame 转成 `BMImage`。
6. YOLO26 adapter 对每帧做 resize/format convert/convert_to。
7. 每次 batch 构造输入 `Tensor` 和 `BMImageArray`。
8. 调用 `engine.process`。
9. 输出转 numpy，做 postprocess 和算法消费。

当前最大浪费点：

- DecodeHub 已经拿到 `BMImage`，但检测侧又回到 numpy，再转回 `BMImage`。
- 每次 infer 都创建输入 `Tensor`、`BMImageArray`、多个中间 `BMImage`。
- batch 内逐帧前处理串行执行。
- 调度日志里的 `total_ms` 被当作“推理时间”，容易掩盖真正 TPU 只忙 6-8ms 的事实。

## 方案选择

选择方案 B：优化检测前处理链路。

理由：

- 当前瓶颈证据明确，preprocess 是总耗时大头。
- 只调配置或降低预览压力不能从根上提高 TPU 忙碌度。
- 9 路 batch 已经启用，继续提高 batch 调度频率会先压垮 CPU/前处理。
- 分阶段减少对象创建和数据搬运，风险可控且收益直接。

## 实施阶段

### 阶段 0：建立基线

目的：确保优化前后有可比较数据。

改动：

- 不改业务逻辑。
- 汇总当前日志里的关键指标。
- 必要时补充更清晰的 metrics 输出，但先避免大改。

记录指标：

- 每个模型 batch=9 的 preprocess/inference/postprocess/total 平均值。
- 每 5 秒 batch scheduler 的 infer 次数、frames 数。
- DecodeHub 每路拉流 fps。
- Display compose fps、MJPEG fanout fps。
- CPU 使用率：decode、display、model_scheduler。
- `bm-smi` TPU 利用率和进程列表。

验证命令：

```bash
curl -s --max-time 2 http://127.0.0.1:5002/api/status | python3 -m json.tool
grep -n "YOLO26 detection time" bm-main.log | tail -30
grep -n "Batch scheduler metrics" bm-main.log | tail -10
grep -n "【拉流】\\|【推流】\\|【推理】" bm-web.log | tail -120
ps -eo pid,ppid,stat,psr,pcpu,pmem,cmd | grep -Ei "python3 main.py|bm-smi" | grep -v grep
bm-smi
```

通过标准：

- 能拿到至少 60 秒稳定运行数据。
- 记录每个模型的 preprocess/inference/postprocess/total 均值。
- 明确当前瓶颈是 preprocess 还是 display。

### 阶段 1：复用 YOLO26 输入对象

目的：减少每次 batch infer 中的对象创建和输入 Tensor 构造开销。

主要文件：

- `core/bm1684x_yolo26_adapter.py`
- 必要时同步 `core/bm1684x_yolov8_adapter.py`，但优先只动 YOLO26。

计划改动：

- 在 detector 初始化时创建可复用的 input tensor。
- 对 batch size 固定的 bmodel，缓存 `BMImageArray{batch}D` 类型和实例。
- 避免每次 `_build_input_tensor` 都创建 `sail.Tensor`。
- 保持最后不足 batch 时仍用最后一帧填充的现有语义。
- `get_last_timing()` 继续输出 preprocess/inference/postprocess/total，便于比较。

风险：

- 复用 Tensor 后要确认 `bmcv.bm_image_to_tensor` 会正确覆盖旧数据。
- 并发调用同一个 detector 时不能共享 mutable buffer；当前每个模型 runtime 在单 scheduler 进程串行调用，风险较低。
- 不同 batch size 的 bmodel 需要兼容。

验证：

```bash
python3 -m py_compile core/bm1684x_yolo26_adapter.py
python3 -m pytest tests/core/test_bm1684x_yolo26_adapter.py -q
```

现场验证：

- 启动 9 路。
- 观察 `YOLO26 detection time`。
- 比较 preprocess 和 total 是否下降。

通过标准：

- 所有现有单元测试通过，或在当前环境缺少 pytest 时至少 py_compile 通过。
- 9 路运行不少于 5 分钟无异常。
- bbox 坐标正常，告警框不明显漂移。
- preprocess 平均下降至少 10%。

### 阶段 2：减少 numpy 到 BMImage 的重复转换

目的：缩短 `DecodeHub -> model_scheduler -> YOLO adapter` 的数据路径。

主要文件：

- `ingestion/decode_hub.py`
- `core/frame_hub.py`
- `app/pipeline/messages.py`
- `app/pipeline/model_scheduler.py`
- `core/bm1684x_yolo26_adapter.py`

候选做法：

1. 保留现有 shared memory 作为兼容路径。
2. 增加检测侧可选的 `BMImage` 快路径。
3. 如果跨进程传递 `BMImage` 不安全，就先在 DecodeHub 侧做检测输入预处理，将预处理后的 tensor 或紧凑 frame 写入检测缓存。

本轮落地做法：

- 保留 shared memory，不改变消费者协议。
- 新增 `bm1684x.detect_frame_max_width/height`，当前配置为 `640x360`，只缩小检测帧，不影响预览帧。
- 继续通过 `coord_width/coord_height` 表达检测坐标系，避免预览和告警框使用原始视频尺寸误缩放。
- 针对当前 SAIL 缺少 `BMImageArray9D` 的事实，使用 `BMImageArray8D` 处理前 8 个 batch 槽位，再用单帧 Tensor 处理第 9 个槽位，减少 fallback 的重复对象创建和单帧转换次数。

优先策略：

- 先不跨进程传递裸 `BMImage`，因为生命周期和序列化风险较高。
- 评估能否在 DecodeHub 进程完成检测输入 resize/convert，再传更小的 320x320 数据。
- 如果必须保持原始坐标，随 payload 带上 `original_width/original_height`、ratio、padding。

风险：

- 坐标映射容易出错。
- 共享内存格式改变会影响多个消费者。
- DecodeHub 已经承担拉流和预览缩放，继续加活可能压 CPU/BMCV。

验证：

- 对同一帧跑旧链路和新链路，比较检测框数量、类别、置信度、坐标误差。
- 坐标误差目标：主要框小于 2 像素或小于图像宽高 0.5%。
- 多路运行时确认共享内存无 stale frame、无通道串帧。

通过标准：

- preprocess 显著下降，目标 30% 以上。
- 检测结果与旧链路基本一致。
- 9 路运行 10 分钟无崩溃、无通道错位。

### 阶段 3：拆分检测和预览资源竞争

目的：避免预览合成拖慢检测前处理，也避免检测优化被预览 CPU 消耗掩盖。

主要文件：

- `core/display_service.py`
- `ingestion/decode_hub.py`
- `config_bm1684x.json`

计划改动：

- 保留当前 `scale_backend=cv2` 的资源隔离策略，避免预览抢 BMCV。
- 增加预览性能档位：
  - `preview_resolution`: 960x540 或 1280x720。
  - `preview_label_budget`: 每路最大标签数。
  - `preview_alert_overlay_level`: full/simple/off。
- 保持检测链路不依赖预览 FPS。

风险：

- 配置项过多会增加维护成本。
- 预览降级可能影响现场观感。

验证：

- 1280x720 和 960x540 两档分别运行。
- 记录 display compose fps 与 model preprocess 是否互相影响。

通过标准：

- 检测吞吐不因打开多个浏览器预览明显下降。
- 预览低帧率时不触发检测侧无意义等待。

### 阶段 4：调度参数再优化

目的：前处理下降后，再提高调度频率，让 TPU 更满。

主要文件：

- `config_bm1684x.json`
- `app/pipeline/model_scheduler.py`

可调参数：

- `bm1684x.detect_emit_fps`
- `batch_scheduler.stream_interval_s`
- `batch_scheduler.model_intervals_s`
- `batch_scheduler.max_models_per_tick`
- `batch_scheduler.preview_backpressure_enabled`

当前值：

- `detect_emit_fps`: 3
- `batch_streams`: 9
- `stream_interval_s`: 0.12
- `helmet_detection`: 0.15
- `fall_detection`: 0.25
- `fight_detection`: 0.25
- `ventilator_equipment`: 0.5
- `max_models_per_tick`: 2
- `preview_backpressure_enabled`: false

调参策略：

- 先固定模型数量和通道数量。
- 每次只改一个参数。
- 每个配置至少跑 3-5 分钟。
- 同时观察模型输出时延、预览 FPS、CPU、TPU。

建议顺序：

1. 如果 preprocess 已明显下降，尝试 `detect_emit_fps=4`。
2. 如果 CPU 仍有余量，尝试 `max_models_per_tick=3`。
3. 如果某些模型更新过密但业务收益低，保持 ventilator 0.5s，优先提高 fall/fight/helmet。
4. 如果预览严重拖累系统，打开 backpressure 或降低预览档位。

通过标准：

- TPU 利用率上升。
- 关键算法结果更新延迟下降。
- CPU 不长期满载。
- 预览仍可接受。

## 验证矩阵

每个阶段至少覆盖以下场景：

| 场景 | 输入 | 检查项 |
| --- | --- | --- |
| 单路 RTSP | 通道 1 | 坐标、报警框、FPS、无崩溃 |
| 九路 RTSP | 通道 1-9 | batch=9、无串帧、无掉进程 |
| 无浏览器预览 | 只跑后端 | 纯检测吞吐 |
| 单浏览器预览 | 打开首页 | display 对检测影响 |
| 多浏览器预览 | 2-3 个页面 | MJPEG fanout 与检测稳定性 |
| 算法全开 | fall/fight/helmet/crowd/ventilator | 模型复用与结果输出 |

## 指标记录模板

每次测试记录：

```text
commit:
branch:
config:
duration:
streams:
enabled_detectors:

YOLO26:
  fall_detection preprocess/inference/postprocess/total:
  fight_detection preprocess/inference/postprocess/total:
  helmet_detection preprocess/inference/postprocess/total:
  ventilator_equipment preprocess/inference/postprocess/total:

Scheduler:
  batches per 5s:
  frames per 5s:
  no_frame skips:
  backpressure skips:

Preview:
  compose_fps:
  mjpeg_fps:

System:
  decode cpu:
  display cpu:
  scheduler cpu:
  bm-smi tpu:
  memory:

Result:
  pass/fail:
  notes:
```

## 回滚策略

每个阶段独立提交。

推荐提交边界：

1. `Measure current preprocessing bottleneck`
2. `Reuse YOLO26 batch input buffers`
3. `Add detection preprocessing fast path`
4. `Separate preview performance profiles`
5. `Tune scheduler after preprocessing optimization`

回滚方式：

```bash
git revert <commit>
```

如果现场运行异常：

1. 先停止服务。
2. 回滚最近阶段提交。
3. 恢复上一个通过验证的配置。
4. 重新启动并跑 5 分钟稳定性检查。

## 优先实施清单

- [x] 阶段 0：收集 60 秒基线日志。
- [x] 阶段 1：YOLO26 input tensor 与 BMImageArray 复用。
- [x] 阶段 1：运行单元测试或语法检查。
- [x] 阶段 1：9 路现场运行 5 分钟。
- [x] 阶段 2：设计检测快路径，先写兼容接口，不删除旧路径。
- [ ] 阶段 2：做旧/新链路检测结果逐帧对比。
- [x] 阶段 3：把预览压力从检测优化中隔离出来。
- [ ] 阶段 4：基于新瓶颈再调调度参数。

## 决策点

阶段 1 完成后根据数据决定下一步：

- 如果 preprocess 下降明显但 TPU 仍低，进入阶段 4 调度参数。
- 如果 preprocess 下降有限，进入阶段 2 减少 numpy/BMImage 往返。
- 如果 display FPS 继续低但检测吞吐改善，单独开预览优化分支。
- 如果检测结果坐标有偏移，暂停性能优化，优先修正坐标映射。
