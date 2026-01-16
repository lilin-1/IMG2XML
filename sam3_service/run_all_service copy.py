import os
import sys
import yaml
import shutil
import argparse
import concurrent.futures
from pathlib import Path
from typing import List, Dict

# Ensure project root on path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3_service.client import Sam3ServicePool
from scripts.merge_xml import run_text_extraction, merge_xml

# Config paths
CONFIG_PATH = ROOT / "config" / "config.yaml"
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

INPUT_DIR = Path(CONFIG["paths"]["input_dir"])
FINAL_DIR = Path(CONFIG["paths"]["final_dir"])
TEMP_DIR = Path(CONFIG["paths"]["temp_dir"])
INITIAL_PROMPTS: List[str] = CONFIG["sam3"].get("initial_prompts", ["rectangle", "icon", "arrow"])

# Endpoints for the service pool (comma separated env or default)
DEFAULT_ENDPOINTS = os.environ.get("SAM3_ENDPOINTS", "http://127.0.0.1:8001")


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


def sam3_via_service(image_path: Path, pool: Sam3ServicePool) -> Path:
    """Call SAM3 HTTP service and save a simple DrawIO XML. Returns XML path."""
    resp = pool.predict(
        image_path=str(image_path),
        prompts=INITIAL_PROMPTS,
        return_masks=False,
        mask_format="rle",
    )
    canvas_w = resp.get("image_size", {}).get("width", 1000)
    canvas_h = resp.get("image_size", {}).get("height", 1000)
    results = resp.get("results", [])

    out_dir = TEMP_DIR / image_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    xml_path = out_dir / "sam3_output.drawio.xml"
    xml_str = _build_simple_drawio_xml(canvas_w, canvas_h, results)
    with open(xml_path, "w", encoding="utf-8") as f:
        f.write(xml_str)
    return xml_path


def process_single_image(image_path: Path, pool: Sam3ServicePool):
    print(f"\n========== 开始处理：{image_path} ==========")
    img_stem = image_path.stem

    sam3_xml_path = sam3_via_service(image_path, pool)
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
        default=DEFAULT_ENDPOINTS,
        help="逗号分隔的 SAM3 服务端地址列表，默认取环境变量 SAM3_ENDPOINTS 或 http://127.0.0.1:8001",
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

    endpoints = [e.strip() for e in args.endpoints.split(",") if e.strip()]
    if not endpoints:
        print("错误：未提供有效的 SAM3 端点，使用 --endpoints 或环境变量 SAM3_ENDPOINTS 配置")
        sys.exit(1)

    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    pool = Sam3ServicePool(endpoints)

    supported = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    images = [p for p in INPUT_DIR.iterdir() if p.suffix.lower() in supported]
    if not images:
        print(f"错误：输入目录{INPUT_DIR}中未找到支持的图片文件")
        sys.exit(1)

    print(f"发现 {len(images)} 张图片，将使用 {args.workers} 线程处理，端点：{endpoints}")

    def _task(img: Path):
        try:
            process_single_image(img, pool)
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
