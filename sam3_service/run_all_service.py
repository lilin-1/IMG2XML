import os
import io
import sys
import yaml
import shutil
import base64
import argparse
import concurrent.futures
from pathlib import Path
from typing import List, Dict
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image
import cv2

# Ensure project root on path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3_service.client import Sam3ServicePool
from sam3_service.rmbg_client import RMBGServicePool
from scripts.merge_xml import run_text_extraction, merge_xml
from scripts.sam3_extractor import (
    extract_style_colors,
    image_to_base64,
    deduplicate_elements,
    build_drawio_xml,
    calculate_element_area,
    RMBGInference,
    VECTOR_SUPPORTED_PROMPTS,
)

# Config paths
CONFIG_PATH = ROOT / "config" / "config.yaml"
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

INPUT_DIR = Path(CONFIG["paths"]["input_dir"])
FINAL_DIR = Path(CONFIG["paths"]["final_dir"])
TEMP_DIR = Path(CONFIG["paths"]["temp_dir"])
INITIAL_PROMPTS: List[str] = CONFIG["sam3"].get("initial_prompts", ["rectangle", "icon", "arrow"])
SCORE_THRESHOLD = CONFIG["sam3"].get("score_threshold", 0.5)
MIN_AREA = CONFIG["sam3"].get("min_area", 100)
RMBG_MODEL_PATH = os.path.join(ROOT, "models", "rmbg", "model.onnx")

# Endpoints for the service pool (argument > env > config)
SAM3_ENDPOINTS_ENV = os.environ.get("SAM3_ENDPOINTS", "")
RMBG_ENDPOINTS_ENV = os.environ.get("RMBG_ENDPOINTS", "")
SAM3_ENDPOINTS_CFG = CONFIG.get("services", {}).get("sam3_endpoints", [])
RMBG_ENDPOINTS_CFG = CONFIG.get("services", {}).get("rmbg_endpoints", [])


def _build_simple_drawio_xml(canvas_w: int, canvas_h: int, results: List[Dict]) -> str:
    """
    Very small DrawIO XML builder to satisfy downstream merge_xml. We render each detection as a rectangle
    with the prompt as the label. This avoids pulling the heavy sam3_extractor module.
    Returns the XML string.
    """
    import xml.etree.ElementTree as ET

    mxfile = ET.Element("mxfile", {"host": "app.diagrams.net", "type": "device"})
    diagram = ET.SubElement(mxfile, "diagram", {"id": "ERDiagram", "name": "Page-1"})
    model = ET.SubElement(
        diagram,
        "mxGraphModel",
        {
            "dx": str(canvas_w),
            "dy": str(canvas_h),
            "grid": "1",
            "gridSize": "10",
            "guides": "1",
            "tooltips": "1",
            "connect": "1",
            "arrows": "1",
            "fold": "1",
            "page": "1",
            "pageScale": "1",
            "pageWidth": str(canvas_w),
            "pageHeight": str(canvas_h),
            "math": "0",
            "shadow": "0",
        },
    )
    root = ET.SubElement(model, "root")
    ET.SubElement(root, "mxCell", {"id": "0"})
    ET.SubElement(root, "mxCell", {"id": "1", "parent": "0"})

    cell_id = 2
    for det in results:
        x1, y1, x2, y2 = det["bbox"]
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)
        label = det.get("prompt", "")
        style = "rounded=0;whiteSpace=wrap;html=1;strokeColor=#000000;fillColor=#ffffff;"  # minimal
        cell = ET.SubElement(
            root,
            "mxCell",
            {"id": str(cell_id), "parent": "1", "vertex": "1", "value": label, "style": style},
        )
        ET.SubElement(
            cell,
            "mxGeometry",
            {"x": str(x1), "y": str(y1), "width": str(w), "height": str(h), "as": "geometry"},
        )
        cell_id += 1

    xml_str = ET.tostring(mxfile, encoding="utf-8")
    return xml_str.decode("utf-8")


def sam3_via_service(image_path: Path, sam3_pool: Sam3ServicePool, rmbg_pool: RMBGServicePool) -> Path:
    """调用 SAM3 HTTP 服务，复用本地后处理（取色/抠图/去重）并生成 DrawIO XML。返回 XML 路径。"""
    resp = sam3_pool.predict(
        image_path=str(image_path),
        prompts=INITIAL_PROMPTS,
        return_masks=True,
        mask_format="png",
    )
    canvas_w = resp.get("image_size", {}).get("width", 1000)
    canvas_h = resp.get("image_size", {}).get("height", 1000)
    detections = resp.get("results", [])

    pil_image = Image.open(image_path).convert("RGB")
    cv2_image = cv2.imread(str(image_path))

    elements_data: Dict[str, List[Dict]] = {}
    for det in detections:
        prompt = det.get("prompt", "")
        score = float(det.get("score", 0))
        if score < SCORE_THRESHOLD:
            continue
        bbox = [int(v) for v in det.get("bbox", [0, 0, 0, 0])]
        x1, y1, x2, y2 = bbox
        if x2 <= x1 or y2 <= y1:
            continue

        area = calculate_element_area(bbox)
        if area < MIN_AREA:
            continue

        polygon = det.get("polygon", []) or []

        elem = {
            "id": len(detections),
            "score": score,
            "bbox": bbox,
            "polygon": polygon,
            "area": area,
        }

        # 生成附加属性
        if prompt == "icon":
            crop = pil_image.crop((x1, y1, x2, y2))
            buf = io.BytesIO()
            crop.save(buf, format="PNG")
            buf.seek(0)
            crop_b64 = base64.b64encode(buf.read()).decode("ascii")
            rgba_b64 = rmbg_pool.remove(crop_b64)
            elem["base64"] = rgba_b64
        elif prompt == "picture":
            crop = pil_image.crop((x1, y1, x2, y2))
            elem["base64"] = image_to_base64(crop)
        elif prompt in VECTOR_SUPPORTED_PROMPTS:
            fill_color, stroke_color = extract_style_colors(cv2_image, bbox)
            elem["fill_color"] = fill_color
            elem["stroke_color"] = stroke_color

        if prompt not in elements_data:
            elements_data[prompt] = []
        elements_data[prompt].append(elem)

    # 去重
    if elements_data:
        elements_data = deduplicate_elements(elements_data, iou_threshold=0.85)

    out_dir = TEMP_DIR / image_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    xml_path = out_dir / "sam3_output.drawio.xml"
    xml_root = build_drawio_xml(canvas_w, canvas_h, elements_data)
    xml_str = ET.tostring(xml_root, encoding="utf-8").decode("utf-8")
    with open(xml_path, "w", encoding="utf-8") as f:
        f.write(xml_str)
    return xml_path


def process_single_image(image_path: Path, sam3_pool: Sam3ServicePool, rmbg_pool: RMBGServicePool):
    print(f"\n========== 开始处理：{image_path} ==========")
    img_stem = image_path.stem

    sam3_xml_path = sam3_via_service(image_path, sam3_pool, rmbg_pool)
    if not sam3_xml_path.exists() or sam3_xml_path.stat().st_size == 0:
        raise RuntimeError(f"SAM3生成的XML为空或不存在：{sam3_xml_path}")

    print("步骤2：文字识别...")
    text_xml_path = run_text_extraction(str(image_path))
    if not os.path.exists(text_xml_path) or os.path.getsize(text_xml_path) == 0:
        raise RuntimeError(f"文字识别生成的XML为空或不存在：{text_xml_path}")

    print("步骤3：合并XML...")
    merged_xml_path = TEMP_DIR / f"{img_stem}_merged_temp.drawio.xml"
    merge_xml(str(sam3_xml_path), text_xml_path, str(image_path), str(merged_xml_path))

    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    final_output_path = FINAL_DIR / f"{img_stem}.drawio.xml"
    shutil.copy(str(merged_xml_path), str(final_output_path))
    print(f"✅ 处理完成，结果已保存: {final_output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run pipeline using SAM3 HTTP service")
    parser.add_argument(
        "--endpoints",
        default="",
        help="逗号分隔的 SAM3 服务端地址列表，优先级：参数 > 环境变量 SAM3_ENDPOINTS > config.services.sam3_endpoints",
    )
    parser.add_argument(
        "--rmbg-endpoints",
        default="",
        help="逗号分隔的 RMBG 服务端地址列表，优先级：参数 > 环境变量 RMBG_ENDPOINTS > config.services.rmbg_endpoints",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="并发处理图片的线程数（建议不要超过服务端总进程数）",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    def _pick_endpoints(arg_val: str, env_val: str, cfg_val: List[str]):
        arg_list = [e.strip() for e in arg_val.split(",") if e.strip()]
        env_list = [e.strip() for e in env_val.split(",") if e.strip()]
        return arg_list or env_list or cfg_val

    endpoints = _pick_endpoints(args.endpoints, SAM3_ENDPOINTS_ENV, SAM3_ENDPOINTS_CFG)
    if not endpoints:
        print("错误：未提供有效的 SAM3 端点，使用 --endpoints、环境变量 SAM3_ENDPOINTS 或 config.services.sam3_endpoints 配置")
        sys.exit(1)

    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    sam3_pool = Sam3ServicePool(endpoints)
    rmbg_endpoints = _pick_endpoints(args.rmbg_endpoints, RMBG_ENDPOINTS_ENV, RMBG_ENDPOINTS_CFG)
    if not rmbg_endpoints:
        print("错误：未提供 RMBG 端点，使用 --rmbg-endpoints、环境变量 RMBG_ENDPOINTS 或 config.services.rmbg_endpoints 配置")
        sys.exit(1)
    rmbg_pool = RMBGServicePool(rmbg_endpoints)

    supported = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    images = [p for p in INPUT_DIR.iterdir() if p.suffix.lower() in supported]
    if not images:
        print(f"错误：输入目录{INPUT_DIR}中未找到支持的图片文件")
        sys.exit(1)

    print(f"发现 {len(images)} 张图片，将使用 {args.workers} 线程处理，SAM3 端点：{endpoints}，RMBG 端点：{rmbg_endpoints}")

    def _task(img: Path):
        try:
            process_single_image(img, sam3_pool, rmbg_pool)
            return True, str(img)
        except Exception as exc:
            print(f"处理 {img} 失败：{exc}")
            return False, str(img)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        results = list(executor.map(_task, images))

    failed = [img for ok, img in results if not ok]
    if failed:
        print(f"\n有 {len(failed)} 张图片处理失败：{failed}")
    else:
        print(f"\n所有图片处理完成！最终文件保存在：{FINAL_DIR}")


if __name__ == "__main__":
    main()
