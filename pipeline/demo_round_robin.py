import itertools
import sys
import time
from pathlib import Path

# Ensure project root on sys.path so sam3_service can be imported when running as a script
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3_service.client import Sam3ServicePool

# 可修改参数
IMAGE = Path("/home/lijianhui/workspace/IMG2XML/pipeline/test.jpg")
ENDPOINTS = [
    "http://127.0.0.1:8001",
    "http://127.0.0.1:8002",
]
PROMPTS = ["rectangle", "arrow"]
NUM_CALLS = 4  # 调用次数，用于展示轮询顺序
RETURN_MASKS = False
MASK_FORMAT = "rle"

def main():
    if not IMAGE.exists():
        raise FileNotFoundError(f"Image not found: {IMAGE}")

    pool = Sam3ServicePool(ENDPOINTS)
    local_cycle = itertools.cycle(range(len(ENDPOINTS)))  # 与池内部相同的轮询顺序（单线程前提）

    for i in range(NUM_CALLS):
        expected_idx = next(local_cycle)
        expected_ep = ENDPOINTS[expected_idx]
        t0 = time.time()
        resp = pool.predict(
            image_path=str(IMAGE),
            prompts=PROMPTS,
            return_masks=RETURN_MASKS,
            mask_format=MASK_FORMAT,
        )
        dt = time.time() - t0
        print(
            f"call={i+1} expected_endpoint={expected_ep} elapsed={dt:.2f}s results={len(resp.get('results', []))}"
        )

    print("done")


if __name__ == "__main__":
    main()
