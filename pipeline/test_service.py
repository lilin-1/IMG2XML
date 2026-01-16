from sam3_service.client import Sam3ServicePool

pool = Sam3ServicePool([
    "http://127.0.0.1:8001",
    "http://127.0.0.1:8002",
])
resp = pool.predict(
    image_path="/home/lijianhui/workspace/IMG2XML/pipeline/test.jpg",
    prompts=["rectangle", "arrow"],
    return_masks=True,
    mask_format="png",
)
print(resp["results"][:2])