# dwa_cpu — 串行 CPU 版 DWA（计时基准）

把 `eval/dwa/planner.py`（向量化 torch DWA）用 C **串行**复刻一遍，用来测量**不同候选轨迹数 NC = v_samples × omega_samples（"扫描条数"）下，单步规划的时间成本**。

torch 版把 NC 个候选 `(v, ω)` 作为一个批量一次算完；C 版逐个候选、逐个障碍点串行扫描。
实现已对拍验证忠实（见下「验证结论」）。

## 文件

| 文件 | 作用 |
|------|------|
| `dwa_cpu.c` | C 串行内核 + 驱动：读快照 + `dwa_config.json`，串行规划，`--grid` 单档 / `--sweep` 多档计时 |
| `bench_dwa.py` | 测时脚本：单进程跑 C `--sweep`，打印各档单步耗时（无 torch 依赖） |
| `bench_onnx.py` | 测导出策略 `policy.onnx` 的单步 CPU 推理延迟（与 DWA 单步成本对比） |
| `build.bat` | 用 MSVC（vswhere 定位 vcvars64）编译 `dwa_cpu.c` |
| `snapshots.bin` | **计时输入夹具**：12288 条真实 DWA 输入（N=105），见下「关于 snapshots.bin」 |

`dwa_cpu.exe` 是构建产物（`build.bat` 重建）。

## 用法

```powershell
# 1) 编译（需要 MSVC Build Tools）
eval\dwa_cpu\build.bat

# 2) 测时：默认档位 3x5 5x9 7x13 9x17 11x21 21x41，相对 9x17 baseline
python -m eval.dwa_cpu.bench_dwa
python -m eval.dwa_cpu.bench_dwa --grids 3x5 9x17 21x41 --repeat 8 --warmup 2

# 3) 对比学习策略的单步延迟
python -m eval.dwa_cpu.bench_onnx
```

`bench_dwa.py` 若发现 `dwa_cpu.exe` 缺失会自动调用 `build.bat`。也可直接跑 C：

```powershell
# 不带 --grid 时读 dwa_config.json 的默认档（当前 baseline = 9x17, NC=153）
eval\dwa_cpu\dwa_cpu.exe eval\dwa_cpu\snapshots.bin eval\dwa_config.json --repeat 5 --out acts.bin
# 单进程多档计时（CPU 状态一致 -> 跨档可比、单调）
eval\dwa_cpu\dwa_cpu.exe eval\dwa_cpu\snapshots.bin eval\dwa_config.json --sweep 3x5,7x13,9x17,11x21,21x41 --repeat 8 --warmup 2
```

输出形如：
```
RESULT grid=9x17 NC=153 N=105 n=12288 repeat=5 warmup=1 per_plan_us=79.58
```

## 计时结果（本机 CPU，串行单线程）

| 档位 | NC | 单步耗时 | plans/s | × baseline |
|------|----|---------|---------|-----------|
| 3×5 | 15 | ~7.1 µs | ~141k/s | 0.09× |
| 5×9 | 45 | ~22.9 µs | ~44k/s | 0.29× |
| 7×13 | 91 | ~47.6 µs | ~21k/s | 0.59× |
| **9×17 (baseline)** | **153** | **~80 µs** | **~12.4k/s** | **1.00×** |
| 11×21 | 231 | ~121 µs | ~8.2k/s | 1.51× |
| 21×41 | 861 | ~452 µs | ~2.2k/s | 5.63× |

成本随 NC 严格单调、线性（≈ 0.5 µs/候选，每候选需扫 3N=315 个障碍点）。

> 参照：同机导出的 cnn_zeropad 策略 `policy.onnx` 单步前向 ≈ 90–100 µs（见 `bench_onnx.py`）。
> baseline DWA 9×17（~80 µs）略快于该学习策略。

## 参数来源（对齐 `DWAConfig.from_configs`）

- **env 派生几何**（`dt, v_min, v_max, omega_max, v_acc_max, omega_acc_max, dist_clip_m`）：
  写在 `snapshots.bin` 头部，C 直接读取。其中 `v_min = 0`（`orientation_verify=true`），
  `v_acc = (v_max−v_min)/dt`、`ω_acc = 2ω_max/dt` —— 故动态窗口 V_d 实际退化为全范围 V_s（与 torch 版一致）。
- **纯 DWA 旋钮**（`predict_time, v_samples, omega_samples, alpha/beta/gamma, robot_radius_m,
  v_brake_acc, omega_brake_acc, rotate_away_mode`）：C 从 `eval/dwa_config.json` 读取；
  `--grid NVxNW` / `--sweep` 覆盖 `v_samples/omega_samples`（计时扫描用）。

## 算法对应（C ↔ planner.py）

逐项复刻：障碍线场（每条 ray → 中心 + 2 个垂向端点 = 3N 点，无效 ray 移到远处幻影）、
动态窗口 V_d、闭式 arc-to-collision（直线/圆弧两分支）、终点位姿的 heading 项、
速度/距离项、按环境归一化（`best/clamp_min(1.0)`）、admissibility V_a（刹车距离判据）、
`argmax`（严格 `>` ⇒ 取首个最大，与 `torch.argmax` 一致）、空集时 `rotate_away` 兜底。

候选展平顺序与 torch 一致：`idx = k*NW + j`，`v=v_grid[k]`、`ω=ω_grid[j]`，保证平局取胜规则相同。

## 验证结论（已完成）

移植正确性曾用一套「torch 参考对拍」脚本在 12288 条真实快照上逐档验证（验证后该脚本已移除）：

- **C(double) vs torch(float64)：不一致率 0.000%**（最大动作差 ~1e-16，机器精度）—— 相同精度下逐位一致，端口忠实。
- **C(double) vs torch(float32)：~9–11%**，且**逐档等于 torch 自身 f32-vs-f64 的不一致率**。
  即这点差异是 **float32 生产实现自身在打分平局处的舍入噪声**，并非 C 端口引入。
  C 版即当前 DWA 的精确双精度实现。

## 关于 snapshots.bin

`snapshots.bin` 是当初用固定种子（seed=0）闭环跑 `EvalEnv` + torch planner 采集的真实 DWA
输入，作为**冻结夹具**供计时复用（计时与具体精度/动作无关，只需代表性的输入分布）。
生成它的导出脚本在验证完成后已删除；若需换种子/规模重新采样，从版本历史或对应会话恢复导出器即可。
