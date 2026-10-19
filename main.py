from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import io
from PIL import Image

app = FastAPI(title="ShrimpTrack API")

# เปิดใช้งาน CORS เพื่ออนุญาตให้หน้าเว็บเชื่อมต่อกับ Backend ได้ทุกช่องทางอย่างอิสระ
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# โหลดโมเดล YOLOv8
model = YOLO("best.pt")

@app.get("/")
def home():
    return {"message": "ShrimpTrack Backend API Running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        results = model(image, conf=0.25)
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = model.names[cls_id]
                detections.append({
                    "stage": class_name,
                    "confidence": round(conf * 100, 2)
                })
                
        return {
            "success": True,
            "total_detected": len(detections),
            "detections": detections
        }
    except Exception as e:
        return {"success": False, "message": str(e)}