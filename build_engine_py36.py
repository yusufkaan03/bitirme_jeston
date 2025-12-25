import tensorrt as trt

ONNX_PATH = "yolo11n-seg.onnx"
ENGINE_PATH = "yolo11n-seg_320_fp16.engine"
FP16 = True
WORKSPACE_GB = 1  # 1GB

TRT_LOGGER = trt.Logger(trt.Logger.INFO)

def build_engine(onnx_path, engine_path, fp16=True, workspace_gb=1):
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            print("[ERROR] ONNX parse failed.")
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            return False

    config = builder.create_builder_config()
    config.max_workspace_size = workspace_gb * (1 << 30)

    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("[INFO] FP16 enabled")
    else:
        print("[INFO] FP16 not enabled (platform_has_fast_fp16:", builder.platform_has_fast_fp16, ")")

    # Build
    print("[INFO] Building engine... (this can take a while)")
    engine = builder.build_engine(network, config)
    if engine is None:
        print("[ERROR] build_engine returned None")
        return False

    with open(engine_path, "wb") as f:
        f.write(engine.serialize())

    print("[OK] Saved engine:", engine_path)
    return True

if __name__ == "__main__":
    ok = build_engine(ONNX_PATH, ENGINE_PATH, FP16, WORKSPACE_GB)
    raise SystemExit(0 if ok else 1)

