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

# ตารางเงื่อนไขระยะพัฒนาการและพยากรณ์วันฟักที่ถูกต้องแม่นยำตามที่คุณกำหนด 100%
STAGE_MAP = {
    "stage_1": {
        "name": "ระยะที่ 1",
        "hatch_forecast": "อีก 21 วัน",
    },
    "stage_2": {
        "name": "ระยะที่ 2",
        "hatch_forecast": "อีก 17 วัน",
    },
    "stage_3": {
        "name": "ระยะที่ 3",
        "hatch_forecast": "อีก 14 วัน",
    },
    "stage_4": {
        "name": "ระยะที่ 4",
        "hatch_forecast": "อีก 11 วัน",
    },
    "stage_5": {
        "name": "ระยะที่ 5",
        "hatch_forecast": "อีก 8 วัน",
    },
    "stage_6": {
        "name": "ระยะที่ 6",
        "hatch_forecast": "อีก 5 วัน",
    },
    "stage_7": {
        "name": "ระยะที่ 7",
        "hatch_forecast": "อีก 2 วัน",
    },
}

# ระบบป้องกัน Cold Start (สะกิดเซิร์ฟเวอร์ไม่ให้หลับ ป้องกันอาการหน่วงเกิน 1 นาที)
async def keep_alive():
  while True:
    await asyncio.sleep(180)
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
  return {"status": "active"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
  try:
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # ปรับสเกล imgsz=320 เพื่อให้ประมวลผลได้อย่างรวดเร็วและแม่นยำสูงสุด
    results = model(image, imgsz=320, conf=0.15, verbose=False)

    detections = []
    for result in results:
      boxes = result.boxes
      for box in boxes:
        cls_id = int(box.cls[0])
        raw_class = model.names[cls_id]
        clean_key = raw_class.lower().strip()

        stage_info = STAGE_MAP.get(
            clean_key,
            {"name": f"ระยะ: {raw_class}", "hatch_forecast": "ไม่พบข้อมูล"},
        )

        detections.append({
            "stage_name": stage_info["name"],
            "hatch_forecast": stage_info["hatch_forecast"],
        })

    return {
        "success": True,
        "total_detected": len(detections),
        "predictions": detections,
    }
  except Exception as e:
    return {"success": False, "message": str(e)}