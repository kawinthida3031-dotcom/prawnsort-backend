from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# โหลดโมเดล YOLOv8
model = YOLO("best.pt")

# ตารางแมปข้อมูลระยะพัฒนาการ 7 ระยะ และพยากรณ์วันฟัก
STAGE_MAP = {
    "stage_1": {"name": "ระยะที่ 1: ไข่เริ่มปฏิสนธิ (สีส้ม/เขียวสด)", "days_left": 14},
    "stage_2": {"name": "ระยะที่ 2: ไข่เริ่มแบ่งเซลล์", "days_left": 12},
    "stage_3": {"name": "ระยะที่ 3: เริ่มปรากฏจุดตา (Eye Spot)", "days_left": 9},
    "stage_4": {"name": "ระยะที่ 4: ตาชัดเจน ตัวเริ่มเป็นรูปทรง", "days_left": 6},
    "stage_5": {"name": "ระยะที่ 5: ตัวอ่อนเคลื่อนไหวในไข่", "days_left": 4},
    "stage_6": {"name": "ระยะที่ 6: ไข่ใสพร้อมฟัก (1-2 วัน)", "days_left": 2},
    "stage_7": {"name": "ระยะที่ 7: ระยะฟักเป็นตัวอ่อน (Hatching)", "days_left": 0},
}

@app.get("/")
def home():
    return {"message": "Prawnsort Backend API Running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # ประมวลผลด้วย YOLO (ตั้งค่า imgsz=640 เพื่อความรวดเร็วและแม่นยำ)
        results = model(image, imgsz=640, conf=0.25)
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                raw_class = model.names[cls_id]
                
                # ดึงข้อมูลระยะและวันฟักจาก STAGE_MAP
                stage_info = STAGE_MAP.get(raw_class, {
                    "name": f"ระยะ: {raw_class}",
                    "days_left": None
                })
                
                days_left = stage_info["days_left"]
                if days_left is None:
                    forecast_str = "ไม่สามารถพยากรณ์ได้"
                elif days_left == 0:
                    forecast_str = "กำลังฟักตัววันนี้ / ภายใน 24 ชม."
                else:
                    forecast_str = f"ประมาณอีก {days_left} วัน"

                detections.append({
                    "class": raw_class,
                    "stage_name": stage_info["name"],
                    "confidence": round(conf * 100, 2),
                    "hatch_forecast": forecast_str
                })
        
        return {
            "success": True,
            "total_detected": len(detections),
            "predictions": detections
        }
    except Exception as e:
        return {"success": False, "message": str(e)}