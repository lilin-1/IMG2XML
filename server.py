import os
import subprocess

def get_freest_gpu():
    """Run nvidia-smi to find the GPU with low utilization and sufficient memory."""
    try:
        # Query GPU index, utilization.gpu, and memory.free
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.free', '--format=csv,noheader,nounits'],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        if result.returncode != 0:
            print(f"⚠️ Warning: nvidia-smi failed, defaulting to GPU 0.")
            return "0"

        gpu_candidates = []
        for line in result.stdout.strip().split('\n'):
            parts = line.split(',')
            if len(parts) == 3:
                index = parts[0].strip()
                utilization = int(parts[1].strip())
                memory_free = int(parts[2].strip())
                gpu_candidates.append({
                    "index": index, 
                    "util": utilization, 
                    "mem": memory_free
                })

        if not gpu_candidates:
            return "0"

        # Auto-select best GPU
        best_gpu = sorted(gpu_candidates, key=lambda x: (x["util"], -x["mem"]))[0]
        
        print(f"✅ Auto-selected GPU {best_gpu['index']} (Util: {best_gpu['util']}%, Free: {best_gpu['mem']} MiB)")
        return best_gpu["index"]

    except Exception:
        return "0"

if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = get_freest_gpu()

import shutil
import uuid
import yaml
import hashlib
import json
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException, Depends
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session # Added
from datetime import datetime # Added

# 导入你的核心处理逻辑
# 确保你的 scripts 目录在 PYTHONPATH 中，或者我们动态添加
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "scripts"))
from scripts.sam3_extractor import main as sam3_main, Sam3ElementExtractor
from scripts.merge_xml import run_text_extraction, merge_xml

# --- 数据库与认证集成 ---
from db.database import engine, get_db
from db import models
from auth import router as auth_router
from auth.auth import get_current_user

# 创建数据库表
models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="SAM3 Image to DrawIO API")

# 注册认证路由
app.include_router(auth_router.router, prefix="/auth", tags=["auth"])

@app.get("/")
def root():
    return {"message": "SAM3 Backend is running correctly. Please access via Frontend (http://localhost:5173)."}

# 允许跨域，方便前端开发
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------- 全局配置与路径管理 --------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "config", "config.yaml")

# 加载配置
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

# Web 任务的存储目录
WEB_TASKS_DIR = os.path.join(BASE_DIR, "output", "web_tasks")
os.makedirs(WEB_TASKS_DIR, exist_ok=True)

class TaskResponse(BaseModel):
    task_id: str
    status: str
    message: str
    cached: bool = False

class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    progress: float  # 0.0 - 1.0
    original_image_url: Optional[str] = None
    result_xml_url: Optional[str] = None
    error: Optional[str] = None
    created_at: Optional[str] = None

import threading

# 全局 SAM3 实例 (Lazy Loaded)
GLOBAL_EXTRACTOR = None
GLOBAL_LOCK = threading.Lock() # 保护 GLOBAL_EXTRACTOR 的使用

def get_extractor():
    global GLOBAL_EXTRACTOR
    if GLOBAL_EXTRACTOR is None:
        print("Initializing Global SAM3 Extractor...")
        GLOBAL_EXTRACTOR = Sam3ElementExtractor()
    return GLOBAL_EXTRACTOR

# -------------------------- 核心处理逻辑 --------------------------
def process_image_background(task_id: str, image_path: str, output_dir: str, file_hash: str = None, api_config: dict = None):
    """后台任务：执行完整的 SAM3 + OCR + Merge 流程"""
    from db.database import SessionLocal
    db = SessionLocal()

    try:
        # 更新任务为 processing
        task = db.query(models.Task).filter(models.Task.id == task_id).first()
        if task:
            task.status = "processing"
            task.progress = 10
            db.commit()
        
        # 1. & 2. 并行执行 SAM3 提取 和 OCR 提取
        print(f"[{task_id}] Starting parallel processing: SAM3 + OCR...")
        
        import concurrent.futures

        def run_sam3_with_global_model(img_path):
            """Helper to run SAM3 using the shared loaded model protected by a lock."""
            extractor = get_extractor()
            # 保护 GPU 推理过程，防止多任务并发导致 OOM
            with GLOBAL_LOCK:
                print(f"[{task_id}] Acquired Global SAM3 Lock. Starting inference...")
                # 关键修改：传入 task_dir 作为输出目录，避免 output/temp/original 冲突
                extractor.iterative_extract(img_path, specific_output_dir=output_dir, api_config=api_config)
                print(f"[{task_id}] Global SAM3 Inference finished. Lock released.")
            
            # 结果现在直接位于 task_dir (即 output_dir)
            return os.path.join(output_dir, "sam3_output.drawio.xml")

        # 使用线程池并行运行
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_sam3 = executor.submit(run_sam3_with_global_model, image_path)
            future_text = executor.submit(run_text_extraction, image_path)
            
            sam3_xml_path = future_sam3.result()
            text_xml_path = future_text.result()

        # 更新进度
        task = db.query(models.Task).filter(models.Task.id == task_id).first()
        if task:
            task.progress = 80
            db.commit()
        
        # 3. 合并 XML
        print(f"[{task_id}] Step 3: Merging XML...")
        final_xml_path = os.path.join(output_dir, "result.drawio.xml")
        merge_xml(sam3_xml_path, text_xml_path, image_path, final_xml_path)
        
        # 更新完成状态
        task = db.query(models.Task).filter(models.Task.id == task_id).first()
        if task:
            task.status = "completed"
            task.progress = 100
            task.result_xml_path = final_xml_path
            db.commit()
        
    except Exception as e:
        print(f"[{task_id}] Error: {str(e)}")
        import traceback
        traceback.print_exc()
        
        task = db.query(models.Task).filter(models.Task.id == task_id).first()
        if task:
            task.status = "failed"
            task.error_message = str(e)
            db.commit()
    finally:
        db.close()

# -------------------------- API 接口 --------------------------


@app.post("/upload", response_model=TaskResponse)
async def upload_image(
    background_tasks: BackgroundTasks, 
    file: UploadFile = File(...), 
    force: bool = False,
    current_user: models.User = Depends(get_current_user), # 必须登录
    db: Session = Depends(get_db)
):
    """
    上传图片并开始处理任务 (需鉴权，消耗积分)
    """
    # 1. 检查积分
    if current_user.credit_balance <= 0:
        raise HTTPException(status_code=402, detail="Insufficient credit balance. Please recharge.")
    
    # 读取所有内容计算哈希
    content = await file.read()
    file_hash = hashlib.sha256(content).hexdigest()
    
    # 2. 检查缓存
    if not force:
        existing_task = db.query(models.Task).filter(
            models.Task.file_hash == file_hash, 
            models.Task.status == "completed"
        ).first()
        
        if existing_task and os.path.exists(existing_task.result_xml_path):
            if existing_task.user_id == current_user.id:
                return {
                    "task_id": existing_task.id,
                    "status": "completed",
                    "message": "Image processed loaded from cache.",
                    "cached": True
                }
    
    # 扣除积分
    current_user.credit_balance -= 1
    db.add(current_user)
    
    task_id = str(uuid.uuid4())
    task_dir = os.path.join(WEB_TASKS_DIR, task_id)
    os.makedirs(task_dir, exist_ok=True)
    
    file_ext = Path(file.filename).suffix or ".jpg"
    image_path = os.path.join(task_dir, f"original{file_ext}")
    
    with open(image_path, "wb") as f:
        f.write(content)
        
    # 创建任务记录
    new_task = models.Task(
        id=task_id,
        user_id=current_user.id,
        status="pending",
        progress=0,
        original_filename=file.filename,
        file_hash=file_hash,
        image_path=image_path,
        created_at=datetime.utcnow()
    )
    db.add(new_task)
    db.commit()
    
    # 构造 API 配置 (BYOK)
    api_config = None
    if current_user.openai_api_key:
        api_config = {
            "api_key": current_user.openai_api_key
        }
        if current_user.openai_base_url:
            api_config["base_url"] = current_user.openai_base_url
            
    # 后台处理
    background_tasks.add_task(process_image_background, task_id, image_path, task_dir, file_hash, api_config=api_config)
    
    return {
        "task_id": task_id,
        "status": "pending",
        "message": "Task queued successfully. Credit deducted.",
        "cached": False
    }

@app.get("/task/{task_id}", response_model=TaskStatusResponse)
def get_task_status(
    task_id: str, 
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    task = db.query(models.Task).filter(models.Task.id == task_id).first()
    
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
        
    # 权限检查：只能看自己的任务
    if task.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to access this task")
        
    response = {
        "task_id": task.id,
        "status": task.status,
        "progress": float(task.progress) / 100.0,
        "error": task.error_message,
        "created_at": str(task.created_at)
    }
    
    if task.image_path and os.path.exists(task.image_path):
         response["original_image_url"] = f"/task/{task_id}/image"
         
    if task.status == "completed" and task.result_xml_path:
        response["result_xml_url"] = f"/task/{task_id}/download"
        
    return response

@app.get("/task/{task_id}/image")
def get_task_image(task_id: str, db: Session = Depends(get_db)):
    task = db.query(models.Task).filter(models.Task.id == task_id).first()
    if not task or not os.path.exists(task.image_path):
        raise HTTPException(status_code=404, detail="Image not found")
        
    return FileResponse(task.image_path)

@app.get("/task/{task_id}/download")
def download_result(task_id: str, db: Session = Depends(get_db)):
    task = db.query(models.Task).filter(models.Task.id == task_id).first()
    if not task or not task.result_xml_path or not os.path.exists(task.result_xml_path):
        raise HTTPException(status_code=404, detail="Result file not found")
        
    return FileResponse(
        task.result_xml_path, 
        media_type="application/xml", 
        filename=f"drawio_export_{task_id}.xml"
    )

@app.get("/my-tasks", response_model=List[dict])
def list_my_tasks(
    skip: int = 0, 
    limit: int = 20, 
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    tasks = db.query(models.Task).filter(models.Task.user_id == current_user.id)\
             .order_by(models.Task.created_at.desc())\
             .offset(skip).limit(limit).all()
             
    results = []
    for t in tasks:
        results.append({
            "task_id": t.id,
            "status": t.status,
            "progress": float(t.progress) / 100.0,
            "created_at": str(t.created_at),
            "original_image_url": f"/task/{t.id}/image",
            "result_xml_url": f"/task/{t.id}/download" if t.status == "completed" else None
        })
    return results


class SegmentationRequest(BaseModel):
    task_id: str
    x: int
    y: int

@app.post("/interactive/segment")
async def segment_at_point(
    req: SegmentationRequest,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """交互式单点分割"""
    task = db.query(models.Task).filter(models.Task.id == req.task_id).first()
    
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
        
    if task.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized")

    image_path = task.image_path
    
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image file not found")
        
    try:
        extractor = get_extractor()
        
        # 加锁以确保单个 GPU 模型实例不会并发调用导致错误
        # 即使我们在 extractor 内部使用 state 字典隔离了数据，PyTorch 模型推理本身在同一 CUDA Stream 下是串行的，
        # 但为了避免 Python 侧的状态混乱，加上 Lock 是最安全的做法。
        # 注意：由于我们在 extractor 内部实现了 cache，这里 switching users 不会触发重新 embedding
        with GLOBAL_LOCK:
            result = extractor.extract_at_point(image_path, (req.x, req.y))
        
        if result is None:
             raise HTTPException(status_code=500, detail="Failed to segment object at location")
             
        return {
            "status": "success",
            "data": result
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # 服务器启动入口
    uvicorn.run(app, host="0.0.0.0", port=8001)
