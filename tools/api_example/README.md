# PPOInference 推理接口使用说明
本接口封装在 `ppo_api/inference.py`，用于在不依赖可视化/训练脚本的情况下，快速独立完成“加载权重 → 读取配置 → 接收米制射线并内部归一化 → 前向推理（动作）”。
- 接口类：`ppo_api.inference.PPOInference`
- 默认配置：`ppo_api/config.json`（包含 `vx_max/vy_max/omega_max/patch_meters/ray_max_gap/num_queries/num_heads`、`ckpt_filename`、`execution_provider` 等关键运行参数；I/O 布局固定为 2 轴动作、7 维姿态尾部）
- 默认权重（未显式传入 `ckpt_path` 时的解析顺序，基于 ONNX 推理）：
  1) config.json 中的 `ckpt_filename`（可写相对/绝对路径，默认 `policy.onnx`）
  2) `ppo_api/policy.onnx`
  3) `ppo_api/latest.onnx`
  4) `ppo_api/final.onnx`
- 执行设备：`execution_provider` 可填 `cpu`（默认，实测最快）、`cuda` 或 `tensorrt`（TensorRTExecutionProvider，会按可用性回落到 CUDA/CPU）。
- ONNX 模型输出要求：优先使用 `action` 张量；若仅包含 `mu`/`log_std`，接口会按训练时的 tanh 限幅进行确定性或采样推理。

## 依赖
- Python 3.8+
- `numpy`
- CPUExecutionProvider：`onnxruntime`（默认建议）
- CUDAExecutionProvider：`onnxruntime-gpu` + 对应的 NVIDIA 驱动/CUDA（会自动回落 CPU）
- TensorrtExecutionProvider：`onnxruntime-gpu`（带 TRT 支持）+ TensorRT 库（会自动回落 CUDA/CPU）

## 快速上手（Python）
```python
import numpy as np
from ppo_api.inference import PPOInference

# 1) 创建推理实例（自动加载 ppo_api/config.json 并按默认顺序解析权重；默认 CPUExecutionProvider，如需 CUDA/TRT 请在 config.json 设 execution_provider）
api = PPOInference()

# 2) 准备一帧射线（米制），长度 R 可变；最大值 ≤ patch_meters（内部会校验+归一化）
R = 256
rays_m = np.random.uniform(0.0, 10.0, size=R).astype(np.float32)

# 3) 前向推理（显式传入 prev_* / prev_prev_* 分量 + task_dist）
action = api.infer(
    rays_m=rays_m,
    sin_ref=0.0,
    cos_ref=1.0,
    prev_vx=0.0,
    prev_omega=0.0,
    prev_prev_vx=0.0,
    prev_prev_omega=0.0,
    task_dist=5.0,  # 当前局部任务点距离（米）；将被归一化为 task_dist/patch_meters
)
print(action)  # numpy 数组，形如 [vx, omega]
```

### 批量推理
```python
B, R = 32, 256
rays_batch = np.random.uniform(0.0, 10.0, size=(B, R)).astype(np.float32)
act_batch = api.infer(
    rays_m=rays_batch,
    sin_ref=0.0,
    cos_ref=1.0,
    prev_vx=0.0,
    prev_omega=0.0,
    prev_prev_vx=0.0,
    prev_prev_omega=0.0,
    task_dist=5.0,
)  # 返回形如 [B, 2]
```

### 关于历史指令与任务距离（必填）
显式传入上一时刻与上上一时刻的控制指令分量（单位：SI）：
`prev_vx, prev_omega` 与 `prev_prev_vx, prev_prev_omega`。接口内部会按 config 中的最大值做归一化，并构造观测中的 7 维尾部特征：
- sin_ref, cos_ref
- prev_vx / vx_max, prev_omega / omega_max
- (prev_vx − prev_prev_vx) / (2·vx_max), (prev_omega − prev_prev_omega) / (2·omega_max)
- task_dist / patch_meters

`task_dist` 需提供当前局部任务点与自身的距离（米），范围 [0, patch_meters]；超界或非数会报错。

### 随机/确定性动作
```python
# 默认 deterministic=True（tanh(mu)*limits）
a_det = api.infer(
    rays_m=rays_m, sin_ref=0.0, cos_ref=1.0,
    prev_vx=0.0, prev_omega=0.0,
    prev_prev_vx=0.0, prev_prev_omega=0.0,
    task_dist=5.0,
    deterministic=True,
)

# 若希望与训练一致的随机策略采样：
a_sto = api.infer(
    rays_m=rays_m, sin_ref=0.0, cos_ref=1.0,
    prev_vx=0.0, prev_omega=0.0,
    prev_prev_vx=0.0, prev_prev_omega=0.0,
    task_dist=5.0,
    deterministic=False,
)
```

### 自定义路径与设备
```python
api = PPOInference(
    config_path="/abs/path/to/config.json",  # 不传则用 ppo_api/config.json
    ckpt_path="/abs/path/to/policy.onnx",   # 不传则按默认顺序解析 ONNX 权重
    device="cuda:0"                         # 可显式覆盖为 CUDAExecutionProvider；也可在 config.json 设置 execution_provider="cuda"/"tensorrt"
)
```

## 输入与严格校验规则
- 射线数量 R：由配置自动计算 `R = ceil((2π * patch_meters) / ray_max_gap)`，即按最大间隔 `ray_max_gap` 在圆周长 `2π·patch_meters` 上均分得到的射线数；若 `patch_meters` 或 `ray_max_gap` ≤ 0 会报错。第 0 条射线需与车体朝向对齐，再按设定的均分间隔铺满圆周。
- 射线输入 `rays_m`：单帧 `[R]` 或批量 `[B,R]`，单位米；需满足 `0 <= rays_m <= patch_meters`，否则报错；内部归一化到 `[0,1]`。
- 姿态尾部：固定 7 维 `[sin_ref, cos_ref, prev_vx/vx_max, prev_omega/omega_max, Δvx/(2·vx_max), Δomega/(2·omega_max), task_dist/patch_meters]`。
- 观测总维度：`R + 7`。
- `sin_ref` / `cos_ref`：标量、形如 `[B]` 或 `[B,1]`；需在 [-1,1]，且 `sin^2 + cos^2 ≈ 1`（容差 0.05）。
- 历史指令：需提供 legacy (`prev_cmd/prev_prev_cmd`) 或新式分量（推荐）。
- 任务距离：`task_dist`（米）必填，范围 [0, patch_meters]。

## 输出
- 返回值：单帧 `[2]` 或批量 `[B,2]`，单位 SI，顺序与 `action_axes` 一致（默认 `[vx, omega]`）。
- 若权重维度与配置不符，会在载入或前向时抛出异常。

## 代码位置
- 接口类：`ppo_api/inference.py`
- 本说明：`ppo_api/README.md`
