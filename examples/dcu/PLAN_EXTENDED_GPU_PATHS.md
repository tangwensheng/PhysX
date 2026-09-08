# PhysX DCU Extended GPU Path Test Plan

目标：逐项验证 rigid body 主路径之外的 GPU 功能。先做 smoke/correctness，再做 DCU/NV 对照和性能 profile。

## 对照原则

- 后续所有 correctness/performance 结论必须以 **DCU 与 NV/A800 同参对照** 为准；包括 scene 参数、body 数量、初始 pose/velocity、mesh/heightfield 尺度、`dt`、gravity、solver/broadphase 配置、材质/反弹系数、sleep/moving 统计阈值、步数、warmup 和计时范围。
- 不允许为了让某一端结果“看起来正常”而单独修改 NV 或 DCU 的场景参数/统计口径。若需要临时 workaround，必须另开配置并标明不能作为同参 correctness 结论。
- 记录项除 PASS/FAIL 外，至少包含：frame time/FPS、moving count、sleeping count 或 moving 判定阈值、height/bounds range、NaN/Inf count、最大线/角速度、异常 body id、首次发散 step、接触/碰撞路径是否触发。
- 之前已经在 DCU 上测过的 SDF、deformable surface/GPU cloth、deformable volume/FEM、PBD particle cloth 等，只能作为 DCU 单边 smoke 记录；后续均需要补齐 NV/A800 同参结果，再判断是 DCU 特有问题、PhysX 上游限制，还是测试场景本身的问题。

## 测试清单

| 优先级 | 路径 | 中文说明 | 第一阶段目标 | 通过标准 |
|---|---|---|---|---|
| P0 | TGS solver | TGS 求解器，Temporal Gauss-Seidel 时间高斯-赛德尔约束求解器 | GPU scene 使用 `PxSolverType::eTGS`，跑 box stack | 100~300 步无崩溃、无 NaN/Inf、无明显飞高/穿地 |
| P0 | articulation | 关节/多连杆系统，例如机械臂、链条、铰链结构 | 创建 `PxArticulationReducedCoordinate` 并加入 GPU scene | articulation 能 add 到 scene，link 姿态有限，模拟稳定 |
| P0 | triangle mesh | 三角网格碰撞，静态复杂地形/模型碰撞体 | cooking 一个静态 triangle mesh，让动态 box 落到上面 | box 最终停在 mesh 上方，无 NaN/Inf、无穿透爆炸 |
| P1 | complex triangle mesh | 复杂三角网格碰撞，多三角形/非平面/复杂接触 | 扩大 mesh 和动态刚体数量 | 接触稳定，和 A800 宏观一致 |
| P1 | SDF | Signed Distance Field，有符号距离场 | cooking 带 SDF 的 mesh/convex，尝试触发 SDF 碰撞路径 | 若 DCU SDF/3D texture 为 stub，记录具体缺失 kernel/API |
| P2 | GPU cloth/FEM | GPU 布料/有限元软体仿真 | 运行 PBD cloth 或 deformable surface/volume 最小场景；已新增 `bench_deformable_surface_smoke` | 能创建、simulate；若失败，定位 cooking、kernel、buffer 或 API 缺失 |

## 执行顺序

1. `bench_tgs_smoke`：验证 TGS scene 创建和短步模拟。
2. `bench_articulation_smoke`：验证 articulation 创建、addScene、simulate/fetch。
3. `bench_trimesh_smoke`：验证 triangle mesh cooking 和 mesh contact。
4. 上述三项通过后，再扩展 DCU/NV CSV 对照。
5. 再进入 SDF 和 cloth/FEM，因为它们更可能依赖未迁移的 GPU-only 数据结构、cooking 参数、3D texture/SDF kernel。
6. 对已经做过 DCU smoke 的 SDF、cloth/FEM/PBD，不直接下最终结论；优先补 NV/A800 同参基线，特别是第 1 个失败 step、height/bounds、moving count 和 contact 触发情况。

## 第一批运行方式

```bash
cd examples/dcu
./build.sh bench_tgs_smoke
./build/bench_tgs_smoke
./build.sh bench_articulation_smoke
./build/bench_articulation_smoke
./build.sh bench_trimesh_smoke
./build/bench_trimesh_smoke
./build.sh bench_complex_trimesh_smoke
./build/bench_complex_trimesh_smoke --cleanexit-wait 15
./build.sh bench_sdf_smoke
./build/bench_sdf_smoke
./build.sh bench_deformable_surface_smoke
./build/bench_deformable_surface_smoke 10 --grid 4 --cleanexit-wait 15
./build.sh bench_deformable_volume_smoke
./build/bench_deformable_volume_smoke 1 --voxels 2 --no-plane --cleanexit-wait 15
./build.sh bench_pbd_cloth_smoke
./build/bench_pbd_cloth_smoke 10 --dim 4 --print-all-steps --cleanexit-wait 15
# 如果 cloth helper 路径在 create/populate 阶段崩溃，先跑更保守的 plain PBD particle buffer：
./build.sh bench_pbd_particles_conservative_smoke
./build/bench_pbd_particles_conservative_smoke 10 --dim 4 --print-all-steps --cleanexit-wait 15
# 或运行低风险矩阵：
./run_pbd_cloth_smoke_matrix.sh
TARGET=bench_pbd_particles_conservative_smoke ./run_pbd_cloth_smoke_matrix.sh
```

## 记录要点

每个 target 记录：

- 是否编译通过。
- 是否能创建 GPU scene。
- 是否能 `simulate/fetchResults`。
- 是否出现 NaN/Inf、穿地、飞高、异常速度。
- 如果失败，记录最后输出和第一个报错 kernel/API。

## 当前执行结果

| 路径 | 状态 | 结果 |
|---|---|---|
| rigid convex / convexmesh DCU-NV 对照 | 待补 NV 同参基线 | 当前 DCU 单边结果看起来未明显发散：convex `Avg frame time=46.12 ms`、`Moving bodies=54/254`、`Height range=[0.5, 5.5]`；convexmesh `Avg frame time=23.15 ms (43 FPS)`、`Moving bodies=46/130`、`Height range=[-0.0, 3.7]`。这些 height range 不像 `-100m` 穿透，moving 比例也比 NV mock 的 `130/130` 更接近稳定状态。但该结论暂时只能说明 DCU 单边 smoke 未见明显异常，不能判定正确。NV 侧已观察到问题：`Moving bodies: 130/130` 全部 moving，convexmesh 出现约 `-100m` 穿透。后续必须在不改变任一端参数、sleep/moving 阈值、heightfield/mesh 尺度、反弹/阻尼参数的前提下，补齐 NV/A800 同参输出，并记录首次发散 step、异常 body id、bounds/velocity/contact 信息。 |
| TGS solver | 功能初步修复；cleanexit 自动化稳定 | 原因是 `HipPhysXGpu::createGpuDynamicsContext` 在 `solverType=eTGS` 时返回 `nullptr`，上层继续使用导致 scene 创建阶段 segfault。已按 NV factory 接入 `PxgTGSDynamicsContext`：DCU 上 `bench_tgs_smoke tgs --print-all-steps --cleanexit-wait 0` 可创建 scene，并完成 180 步、64 bodies，`Height range=[0.499981,7.493330]`、`Bad bodies=0`、`VERDICT=PASS`、kernel launches=11068。已在 `bench_tgs_smoke.cpp` 增加 cleanexit 路径；`--cleanexit-wait 0` 下显式 release 后调用 `std::_Exit()`，可稳定返回自动化退出码。底层 release/teardown/静态析构根因后续单独定位。 |
| GPU broadphase forced eGPU | 初步修复，待更大规模回归 | 已将 `BP_COMPUTE_ACTIVE_HISTOGRAM` 在 `PX_DCU_PORT` 下从 512 threads 降到 256，并把 `computeStartAndActiveRegionHistogram` 内部 active histogram warp 数从硬编码 16 改为由 blockDim/WARP_SIZE 推导。DCU 上 `bench_tgs_smoke pgs --force-gpu-bp` 通过，`bench_trimesh_smoke 60/240 --force-gpu-bp` 通过：240 步 `Boxes=16`、`Height range=[0.377173,0.839057]`、`Max speed=0`、`Bad boxes=0`、`VERDICT=PASS`。`Max speed=0` 是最终统计时 boxes 已静止/睡眠的结果，不代表模拟未运行；后续可补逐步 moving/maxSpeed 或更大 body 数回归。 |
| articulation | CLOSED / PASS；普通进程 teardown 单独归档 | `bench_articulation_smoke` 已增强为检查全部 5 个 link 的 pose/velocity、joint position/velocity、首个异常 step、PhysX error、drive response、joint limit 和 settled state，并补齐 `art->release()` 及完整对象释放。PGS/TGS 静态 180 步均 PASS；交替正负 `0.35 rad` drive target 的动态 180 步均产生真实非零响应并收敛。最终 PGS/TGS 300 步均为 `Drive response=PASS`、`Joint limits=PASS`、`Settled state=PASS`、`Bad links=0`、`PhysX fatal=0`、`VERDICT=PASS`；cleanexit 均 exit=0。PGS/TGS normalexit 的物理指标和显式 release 与 cleanexit 完全一致，但都在 `Release finished.` 后的进程级/静态析构阶段 core dump、exit=139；该问题归入通用 DCU/HIP/HSA teardown 技术债，`std::_Exit()` 仅用于自动化隔离。专项记录见 `docs/physx_dcu_articulation_p0_report_2026-08-04.md`。 |
| triangle mesh | 通过 | `bench_trimesh_smoke` 默认 240 步通过：16 boxes 高度范围 `[0.377173, 0.839057]`，`Bad boxes=0`，`VERDICT: PASS`。随后在退出/清理阶段 segfault，已加 `--cleanexit-wait N` 做显式 release 后等待并退出。 |
| complex triangle mesh | 暂停，设备稳定性失败 | `bench_complex_trimesh_smoke` grid=48、8 boxes、60 步模拟统计看似通过：`Height range=[-0.015911,0.511670]`、`Max speed=2.446434`、`Bad boxes=0`、`Below terrain boxes=0`、`VERDICT=PASS`。但随后在显式 release 完成并进入 `--cleanexit-wait 15` 等待时进程被 kill，exit code 137；用户确认该程序会稳定把 DCU 卡弄掉，不是卡先掉导致程序崩。因此该路径不能视为通过，应记录为 simulation result PASS、device stability FAIL。默认 256 boxes、720 步此前在前几步触发 VMFault。已新增 `run_complex_trimesh_smoke_matrix.sh`，但目前暂停继续扩大规模；后续若恢复，应先做最小触发条件和 release/teardown 定位，而不是继续调物理参数。 |
| SDF | CLOSED / PASS | gfx936 SDF trimesh-plane 主链路已闭环：修复 CUDA 32-lane logical warp 兼容、midphase/contact/solver 边界问题，并将 batch>1 多 workgroup 场景中不稳定的第一次通用 `contactReduce` 替换为确定性的 DCU reducer。默认 batch=1 的 300-step/8-body 稳定基线 PASS；实验 batch=2 的 300-step/8-body、五次重复、32-body 压力测试均由用户确认 PASS；`trimeshCollision.cu` 恢复正常内联后的 batch=1/batch=2 消融均 PASS；batch=4 也由用户确认 PASS。临时 Stage/device/host 诊断已清理，远程 clean 重编与最终复验通过。默认 batch 保持 1，batch=2/4 仍为实验路径。NV/A800 同参和性能 profiling 为项目级补充项，不阻塞 SDF 功能关闭。专项记录见 `docs/physx_dcu_sdf_batch2_report_2026-08-04.md`。 |
| deformable surface / GPU cloth | 失败，待 NV 同参确认 | `bench_deformable_surface_smoke` grid=8 可完成 GPU scene 创建、mesh cooking、`createDeformableSurface`、`addActor`、host mirror 分配和 `copyToDevice`；第 1 次 `scene->simulate()` 内 segfault，尚未到 `fetchResults()`。`--no-plane`、`--no-self-collision`、`grid=4` 仍崩。当前结论仅限 DCU：setup/copyToDevice PASS，deformable surface simulate 本体 FAIL。后续需补 NV/A800 同参，确认 NV 是否能进入/完成 simulate，以及 deformable surface GPU 路径是否需要额外上游配置。 |
| deformable volume / FEM | 失败，待 NV 同参确认 | `bench_deformable_volume_smoke 60 --voxels 4 --print-all-steps` 默认 plane y=0 会 VMFault；一次复现为 Step 32 bounds `min.y=-0.52 max.y=0.52`，volume 已跨过/嵌入 plane，随后 Step 33 `simulate()` 内 `Invalid address access`/VMFault。`--no-plane` 和 `--plane-y -100` 不崩。当前 DCU 结论：不是存在 plane actor 即崩，而是 deformable-volume 与 rigid plane 实际接触/穿入后，contact/pair/update/solve GPU 路径非法访问。后续需补 NV/A800 同参，确认 NV 在同一 bounds/contact step 是否稳定，还是场景本身也会失败。 |
| PBD particle cloth / PBD particle buffer | 防崩完成，明确 UNSUPPORTED；真实后端待实现 | `bench_pbd_cloth_smoke` 最小 cloth helper 路径在 `ExtGpu::PxCreateAndPopulateParticleClothBuffer` 阶段 segfault；进一步拆到 plain `PxParticleBuffer` 后确认根因：`HipPhysXGpu::createParticleBuffer/createParticleClothBuffer/createParticleAndDiffuseBuffer/createParticleRigidBuffer` 仍是 stub 并返回 `nullptr`，而 `NpParticleBuffer.cpp` 构造函数直接 `mGpuBuffer->getUniqueId()` 解引用。已在 `NpParticleBuffer.cpp` 为四类 particle buffer 增加判空和 `eINVALID_OPERATION` 错误；在 `NpFactory.cpp` 中检查 low-level buffer，失败时销毁 wrapper 并返回 `NULL`；在 `NpParticleBuffer.h` 增加 tracking guard，避免未加入 tracking 的临时 wrapper 析构时触发 `pure virtual method called`；在 `bench_pbd_particles_conservative_smoke.cpp` 中将 null buffer 判为 `VERDICT: UNSUPPORTED` 并继续完整 release。DCU 验证：`createParticleBuffer returned (nil)` 后输出 `VERDICT: UNSUPPORTED - createParticleBuffer returned null`，随后 `Releasing scene/dispatcher/GPU manager/physics/foundation` 完成，`Total kernel launches=0`，不再 segfault/abort。结论：当前 DCU PBD particle/cloth 真实 GPU buffer 后端仍未实现，但已从空指针崩溃修成明确 unsupported。 |

## 下一任务建议

articulation P0 已完成功能收口。下一项优先建议处理 deformable volume 与 rigid plane 实际接触后约 step 33 的 VMFault：该故障已有 `--no-plane` 和远离 plane 不崩的对照，触发边界明确，适合继续采用 block-uniform stage return 和低侵入诊断逐段定位 contact/pair/update/solve 路径。

complex triangle mesh 继续保持暂停，因为现有最小场景仍有整卡稳定性风险。PBD particle/cloth 属于缺失真实 DCU buffer 后端，需要按功能实现任务单独规划。普通进程退出 exit=139 则作为跨 smoke 的通用 DCU/HIP/HSA teardown 专项，不与单个物理功能路径混合处理。

## 已知背景

之前知识库只列过粗待办：TGS 场景创建崩溃、关节约束未测试、Articulation 未测试、三角网格/Heightfield/布料内核注册待处理。本计划将其拆成可执行 target 和通过标准。

## 本轮修复汇总（提交前）

| 修复点 | 原因 | 修改方法 | 验证结果 | 涉及文件 |
|---|---|---|---|---|
| GPU broadphase forced eGPU | CUDA 默认 `BP_COMPUTE_ACTIVE_HISTOGRAM=512`，但 DCU/HIP 当前该 kernel launch bound 为 256；运行时以 512 threads/block 启动会触发 launch bound/VMFault。 | `PX_DCU_PORT` 下将 `BP_COMPUTE_ACTIVE_HISTOGRAM` 降为 256；kernel 内 active histogram warp 数由 `BP_COMPUTE_ACTIVE_HISTOGRAM / WARP_SIZE` 推导，CUDA/NV 保持 512/16 warp，DCU 为 256/8 warp；逻辑 `WARP_SIZE` 仍保持 32。 | `bench_tgs_smoke pgs --force-gpu-bp` PASS；`bench_trimesh_smoke 60/240 --force-gpu-bp` PASS，240 步 `Bad boxes=0`、`VERDICT=PASS`。 | `physx/source/gpubroadphase/include/PxgBroadPhaseKernelIndices.h`；`physx/source/gpubroadphase/src/CUDA/broadphase.cu` |
| GPU TGS scene 创建 | DCU factory `HipPhysXGpu::createGpuDynamicsContext` 在 `solverType=eTGS` 时返回 `nullptr`，上层继续使用导致 scene 创建阶段 segfault。 | 按 NV `PxgPhysXGpu` factory 接入真实 `PxgTGSDynamicsContext`；PGS 继续使用 `PxgDynamicsContext`。 | `bench_tgs_smoke tgs --print-all-steps --cleanexit-wait 0` 可创建 scene 并完成 180 步、64 bodies，`Bad bodies=0`、`VERDICT=PASS`；`VERDICT` 后退出阶段已由 cleanexit workaround 稳定自动化；底层 teardown 根因单独归档。 | `physx/source/cudamanager/src/HipPhysXGpu.cpp` |
| TGS smoke cleanexit 自动化退出 | TGS `VERDICT: PASS` 后 `main()` 返回阶段可能触发全局/静态析构或 HIP teardown segfault，影响自动化退出码。 | 在 `bench_tgs_smoke.cpp` 增加 `--cleanexit/--cleanexit-wait N`、release 阶段日志和 `std::_Exit(pass ? 0 : 1)`；显式释放 PhysX 对象后跳过进程级析构。 | `bench_tgs_smoke tgs --print-all-steps --cleanexit-wait 0` 已确认稳定退出。 | `examples/dcu/bench_tgs_smoke.cpp` |
| PBD particle buffer 防崩 | DCU particle buffer factory API 仍返回 `nullptr`，上层 `NpParticleBuffer` 未判空并解引用 `mGpuBuffer->getUniqueId()`；factory 销毁未 tracking wrapper 时还可能触发 `pure virtual method called`。 | 四类 particle buffer 构造函数判空并报 `eINVALID_OPERATION`；factory 检查 low-level buffer，失败则销毁 wrapper 并返回 `NULL`；`NpParticleBufferBase` 增加 tracking guard；smoke 将 null buffer 判为 `VERDICT: UNSUPPORTED` 并完整 release。 | `bench_pbd_particles_conservative_smoke 1 --dim 2 --skip-buffer --print-all-steps --cleanexit-wait 0` 输出 `VERDICT: UNSUPPORTED - createParticleBuffer returned null`，完整 release，不再 segfault/abort。 | `physx/source/physx/src/NpParticleBuffer.cpp`；`physx/source/physx/src/NpParticleBuffer.h`；`physx/source/physx/src/NpFactory.cpp`；`examples/dcu/bench_pbd_particles_conservative_smoke.cpp` |

### 提交建议

- 本轮提交只纳入上表相关文件；工作区中存在其他历史/无关改动，提交前需要用路径白名单生成 diff 或 staged changes。
- 建议提交标题：`DCU: fix GPU broadphase/TGS setup and guard unsupported PBD buffers`。
- 退出阶段 `Segmentation fault` 暂不纳入本轮修复，单独记录为 DCU teardown/exit stability。
