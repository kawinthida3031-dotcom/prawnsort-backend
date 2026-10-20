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

# ตารางแมประยะพัฒนาการกุ้งฝอยและพยากรณ์วันฟัก
STAGE_MAP = {
    "stage_1": {
        "name": "ระยะที่ 1: ไข่เริ่มปฏิสนธิ (สีส้ม/เขียวสด)",
        "days_left": 14,
    },
    "stage_2": {"name": "ระยะที่ 2: ไข่เริ่มแบ่งเซลล์", "days_left": 12},
    "stage_3": {
        "name": "ระยะที่ 3: เริ่มปรากฏจุดตา (Eye Spot)",
        "days_left": 9,
    },
    "stage_4": {
        "name": "ระยะที่ 4: ตาชัดเจน ตัวเริ่มเป็นรูปทรง",
        "days_left": 6,
    },
    "stage_5": {"name": "ระยะที่ 5: ตัวอ่อนเคลื่อนไหวในไข่", "days_left": 4},
    "stage_6": {"name": "ระยะที่ 6: ไข่ใสพร้อมฟัก (1-2 วัน)", "days_left": 2},
    "stage_7": {
        "name": "ระยะที่ 7: ระยะฟักเป็นตัวอ่อน (Hatching)",
        "days_left": 0,
    },
}


# ฟังก์ชันป้องกัน Render Sleep (Self-Keep-Alive)
async def keep_alive():
  while True:
    await asyncio.sleep(240)  # สะกิดตัวเองทุกๆ 4 นาที
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
  return {"status": "active", "server_time": time.time()}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
  try:
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # ประมวลผลด่วนด้วยขนาดภาพสเกล 416px
    results = model(image, imgsz=416, conf=0.15, verbose=False)

    detections = []
    for result in results:
      boxes = result.boxes
      for box in boxes:
        cls_id = int(box.cls[0])
        raw_class = model.names[cls_id]
        clean_key = raw_class.lower().strip()

        stage_info = STAGE_MAP.get(
            clean_key, {"name": f"ระยะ: {raw_class}", "days_left": None}
        )

        days_left = stage_info["days_left"]
        if days_left is None:
          forecast_str = "ไม่พบข้อมูล"
        elif days_left == 0:
          forecast_str = "ฟักตัววันนี้ / ภายใน 24 ชม."
        else:
          forecast_str = f"อีก {days_left} วัน"

        detections.append({
            "stage_name": stage_info["name"],
            "hatch_forecast": forecast_str,
        })

    return {
      "success": True,
      "total_detected": len(detections),
      "predictions": detections,
    }
  except Exception as e:
    return {"success": False, "message": str(e)}