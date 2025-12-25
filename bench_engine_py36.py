import time
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401

ENGINE_PATH = "yolo11n-seg_320_fp16.engine"  # dosya adını kendine göre düzelt
WARMUP = 50
ITERS = 200

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def load_engine(path):
    with open(path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def allocate_io(engine, context):
    inputs, outputs, bindings = [], [], []
    stream = cuda.Stream()

    for i in range(engine.num_bindings):
        name = engine.get_binding_name(i)
        dtype = trt.nptype(engine.get_binding_dtype(i))
        shape = context.get_binding_shape(i)

        # Dynamic shape varsa burada sabitle (çoğu export'ta sabit olur)
        if -1 in shape:
            raise RuntimeError(f"Dynamic shape detected for {name}: {shape}")

        size = int(np.prod(shape))
        host = cuda.pagelocked_empty(size, dtype)
        device = cuda.mem_alloc(host.nbytes)
        bindings.append(int(device))

        item = {"name": name, "shape": shape, "dtype": dtype, "host": host, "device": device}
        if engine.binding_is_input(i):
            inputs.append(item)
        else:
            outputs.append(item)

    return inputs, outputs, bindings, stream

def main():
    engine = load_engine(ENGINE_PATH)
    context = engine.create_execution_context()

    # Eğer input shape sabitse context zaten bilir; yine de garanti:
    for i in range(engine.num_bindings):
        if engine.binding_is_input(i):
            shape = engine.get_binding_shape(i)
            context.set_binding_shape(i, shape)

    inputs, outputs, bindings, stream = allocate_io(engine, context)

    # Dummy input (0..1)
    for inp in inputs:
        inp["host"][:] = np.random.rand(inp["host"].size).astype(inp["dtype"]).ravel()

    # Warmup
    for _ in range(WARMUP):
        for inp in inputs:
            cuda.memcpy_htod_async(inp["device"], inp["host"], stream)
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        for out in outputs:
            cuda.memcpy_dtoh_async(out["host"], out["device"], stream)
        stream.synchronize()

    # Timed
    times = []
    for _ in range(ITERS):
        t0 = time.perf_counter()
        for inp in inputs:
            cuda.memcpy_htod_async(inp["device"], inp["host"], stream)
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        for out in outputs:
            cuda.memcpy_dtoh_async(out["host"], out["device"], stream)
        stream.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)

    avg = sum(times) / len(times)
    p50 = sorted(times)[len(times)//2]
    p95 = sorted(times)[int(len(times)*0.95)]
    fps = 1000.0 / avg

    print(f"[TRT] avg={avg:.2f} ms  p50={p50:.2f} ms  p95={p95:.2f} ms  FPS~{fps:.2f}")
    print("Inputs:")
    for x in inputs:
        print(" ", x["name"], x["shape"], x["dtype"])
    print("Outputs:")
    for x in outputs:
        print(" ", x["name"], x["shape"], x["dtype"])

if __name__ == "__main__":
    main()

