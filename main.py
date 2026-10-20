import asyncio
import io
import time
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import httpx
from PIL import Image
from ultralytics import YOLO

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# โหลดโมเดล YOLO
model = YOLO("best.pt")

# ตารางเงื่อนไขพยากรณ์วันฟัก (ล็อกถูกต้อง 100% ไม่มีอธิบายลักษณะไข่)
STAGE_MAP = {
    "stage_1": {"name": "ระยะที่ 1", "hatch_forecast": "อีก 21 วัน"},
    "stage_2": {"name": "ระยะที่ 2", "hatch_forecast": "อีก 17 วัน"},
    "stage_3": {"name": "ระยะที่ 3", "hatch_forecast": "อีก 14 วัน"},
    "stage_4": {"name": "ระยะที่ 4", "hatch_forecast": "อีก 11 วัน"},
    "stage_5": {"name": "ระยะที่ 5", "hatch_forecast": "อีก 8 วัน"},
    "stage_6": {"name": "ระยะที่ 6", "hatch_forecast": "อีก 5 วัน"},
    "stage_7": {"name": "ระยะที่ 7", "hatch_forecast": "อีก 2 วัน"},
}

# ป้องกัน Render Cold Start (สะกิดเซิร์ฟเวอร์ทุก 2 นาที)
async def keep_alive():
    while True:
        await asyncio.sleep(120)
        try:
            async with httpx.AsyncClient() as client:
                await client.get("https://prawnsort-backend.onrender.com/ping")
        except Exception:
            pass

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(keep_alive())

@app.get("/")
@app.get("/ping")
def ping():
    return {"status": "active", "time": time.time()}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # ประมวลผลภาพขนาดเล็ก 320px เพื่อความรวดเร็วสูงสุด
        results = model(image, imgsz=320, conf=0.20, verbose=False)
        
        best_detection = None
        highest_conf = 0.0

        # ล็อกคัดเลือกเฉพาะกุ้งตัวหลักที่เด่นที่สุดเพียงตัวเดียวเท่านั้น (ตัดเงาสะท้อนออก)
        for result in results:
            boxes = result.boxes
            for box in boxes:
                conf = float(box.conf[0])
                if conf > highest_conf:
                    highest_conf = conf
                    cls_id = int(box.cls[0])
                    raw_class = model.names[cls_id]
                    clean_key = raw_class.lower().strip()
                    
                    stage_info = STAGE_MAP.get(clean_key, {
                        "name": f"ระยะ: {raw_class}",
                        "hatch_forecast": "ไม่พบข้อมูล"
                    })
                    
                    best_detection = {
                        "stage_name": stage_info["name"],
                        "hatch_forecast": stage_info["hatch_forecast"]
                    }

        detections = [best_detection] if best_detection else []

        return {
            "success": True,
            "total_detected": len(detections),
            "predictions": detections
        }
    except Exception as e:
        return {"success": False, "message": str(e)}