# Standalone DCU Host-Atomic Reproducer

These two HIP programs do not include or link PhysX.

- `host_mapped_atomic_drop_repro.hip` reproduces the old path: a DCU kernel
  loads the candidate value from a `hipMalloc` device buffer, then executes
  `atomicMax` against memory allocated by `hipHostMalloc(...Mapped)`.
  On the affected gfx936 platform this may produce `UR_ATOMIC_OPCODE`, kill the
  process with exit 137, and reset the selected device.
- `device_atomic_readback_fixed.hip` demonstrates the repair: the candidate
  still comes from the same kind of `hipMalloc` device buffer, but the atomic
  target is also `hipMalloc` device memory. An asynchronous DtoH copy then
  transfers the result into a pinned host readback slot.

The operation corresponding to `convexMesh.cu:762` is `atomicMax`, although it
is sometimes referred to as the atomic-add site during discussion. The key
property is the memory placement of its two operands, not the reduction opcode.

## Build

```bash
cd /public/home/tangwsh/PhysX/examples/dcu/host_atomic_minimal
bash build.sh
```

Set `DCU_ARCH=gfx938` when building for gfx938. The default is gfx936.

## Run the fixed case first

`HIP_VISIBLE_DEVICES=1` selects physical HCU[1]. Inside the process it is exposed
as logical HIP device 0.

```bash
HIP_VISIBLE_DEVICES=1 ./build/device_atomic_readback_fixed
```

Expected output ends with:

```text
[FIXED] readback=1234 expected=1234 verdict=PASS
```

## Run both cases with evidence capture

The runner always rebuilds and runs the fixed case first. It runs the dangerous
case only when the fixed result is `FIXED_STABLE` and the explicit confirmation
is present.

Fixed case only:

```bash
TARGET_HIP_DEVICE=1 bash run_cases.sh
```

Fixed case followed by the dangerous case:

```bash
RUN_DANGEROUS=I_UNDERSTAND \
TARGET_HIP_DEVICE=1 \
HOLD_SECONDS=20 \
bash run_cases.sh
```

The timestamped log is written under `build/host_atomic_cases_*.log`. It records
the build gate, `hy-smi` health, process exits, output markers, and relevant
`dmesg` lines in separate time windows for the fixed and dangerous phases.

## Run the intentional device-drop reproducer

Warning: use an expendable test device. The run can kill the process and reset
the selected DCU. The explicit confirmation variable prevents accidental launch.

```bash
DCU_ALLOW_HOST_ATOMIC_DROP=I_UNDERSTAND \
HIP_VISIBLE_DEVICES=1 \
./build/host_mapped_atomic_drop_repro 20
```

The argument is the post-synchronize hold time in seconds. On the affected
platform, the characteristic result is:

```text
[REPRO] sync=0 (no error) phase=hold seconds=20
Killed
```

The shell normally reports exit 137. Check the kernel log immediately afterward:

```bash
dmesg -T | grep -E 'UR_ATOMIC_OPCODE|atomic err|hycu' | tail -n 20
```

The decisive signature is:

```text
hycu <BDF>: hycu: Found atomic err(0x1) in pcie:UR_ATOMIC_OPCODE
```

Some driver versions reject the transaction without immediately killing the
process. That partial reproduction looks like this:

```text
[REPRO] sync=0 (no error) phase=hold seconds=20
[REPRO] survived hold counter=0 expected=1234
[REPRO] verdict=ATOMIC_REJECTED_WITHOUT_DEVICE_DROP
```

It returns exit 5. This proves that the mapped-host atomic did not commit, but it
is not a complete device-drop reproduction. Preserve the surrounding dmesg lines;
a generic `GCEA err` line alone is weaker evidence than `UR_ATOMIC_OPCODE`.

## Data-flow difference

```text
Before the PhysX fix / reproducer:
DCU kernel
  -> reads candidate value from DCU device memory
  -> issues atomicMax to the GPU mapping of CPU pinned RAM
  -> request leaves the DCU as a PCIe AtomicOp

After the PhysX fix / fixed case:
DCU kernel
  -> reads candidate value from DCU device memory
  -> issues atomicMax to a counter in DCU device memory
  -> normal DtoH copy writes the final value to CPU pinned RAM
```

In the PhysX expression:

```cpp
atomicMax(maxTempMemRequirement,
    calculateConvexMeshPairMemRequirement() * (*midPhasePairsNeeded)
        + calculateAdditionalPadding(nbContactManagers));
```

`midPhasePairsNeeded` is read by the kernel from GPU memory. Before the fix,
`maxTempMemRequirement` pointed at CPU host-mapped memory. After the fix, it
points at `mMaxConvexMeshTempMemoryOnDevice`; the CPU-visible
`mMaxConvexMeshTempMemory` is only the destination of `memcpyDtoHAsync`.
