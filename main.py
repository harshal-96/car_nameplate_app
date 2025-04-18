from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
import easyocr
import io
import base64
import requests
from bs4 import BeautifulSoup
import os
import uvicorn

# Define the FastAPI app
app = FastAPI(
    title="Car Classification & License Plate Detection API",
    description="API for car view classification and license plate detection",
    version="1.0.0"
)

# Set up CORS to allow requests from frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (HTML, CSS, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Define model architecture
class CarSegmentationModel(nn.Module):
    def __init__(self, num_classes=3):
        super(CarSegmentationModel, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# Global variables for models
model = None
device = None
reader = None

# Initialize models on startup
@app.on_event("startup")
async def startup_event():
    global model, device, reader
    # Initialize device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model
    model = CarSegmentationModel(num_classes=3).to(device)
    
    # In a real app, you'd load your trained model here
    # For demo purposes, we'll assume the model is available
    # model.load_state_dict(torch.load("car_segmentation_model.pth", map_location=device))
    
    # For demo purposes, let's just use the pretrained model
    model.eval()
    
    # Initialize OCR reader
    reader = easyocr.Reader(['en'])

# Helper functions
def predict_car_view(image, model, device):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
    
    class_labels = ["back", "front", "side"]
    return class_labels[predicted_class]

def detect_license_plate(image, reader):
    # Convert PIL Image to OpenCV format
    img_array = np.array(image)
    img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Convert to grayscale for better OCR
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    # Perform OCR
    results = reader.readtext(gray)
    
    # Prepare results
    extracted_text = []
    probabilities = {}
    
    # Process results
    for (bbox, text, prob) in results:
        cleaned_text = "".join(char.upper() for char in text if char.isalnum())
        
        if cleaned_text:
            extracted_text.append(cleaned_text)
            probabilities[cleaned_text] = prob
            
            # Get coordinates for drawing
            (top_left, top_right, bottom_right, bottom_left) = bbox
            top_left = tuple(map(int, top_left))
            bottom_right = tuple(map(int, bottom_right))
            
            # Draw rectangle and text
            cv2.rectangle(img_cv, top_left, bottom_right, (0, 255, 0), 2)
            cv2.putText(img_cv, cleaned_text, (top_left[0], top_left[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Convert back to RGB for display
    img_with_detections = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    
    # Convert processed image to base64 for frontend display
    _, buffer = cv2.imencode('.jpg', img_with_detections)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    
    # Filter potential plates
    potential_plates = [text for text in extracted_text if 6 <= len(text) <= 12]
    
    # Find best plate
    best_plate = None
    if potential_plates:
        best_plate = max(potential_plates, key=lambda p: probabilities[p])
        best_plate = best_plate[:10]  # Limit to 10 characters
    
    return img_base64, best_plate

def correct_text(text, expected_type):
    correction_dict = {
        '0': 'O', '1': 'I', '2': 'Z', '3': 'B', '4': 'L', '5': 'S', '6': 'G', '7': 'T', '8': 'B', '9': 'P',
        'O': '0', 'I': '1', 'Z': '2', 'B': '3', 'L': '4', 'S': '5', 'G': '6', 'T': '7', 'B': '8', 'P': '9'
    }
    
    corrected_text = ""
    for char in text:
        if expected_type == "alpha" and char.isdigit():
            corrected_text += correction_dict.get(char, char)
        elif expected_type == "numeric" and char.isalpha():
            corrected_text += correction_dict.get(char, char)
        else:
            corrected_text += char
    return corrected_text

def strict_split_number_plate(number_plate):
    if len(number_plate) < 8:
        return None, None, None, None
    
    # Extract parts
    part1 = number_plate[:2]
    part2 = number_plate[2:4]
    part4 = number_plate[-4:]
    part3 = number_plate[-6:-4] if len(number_plate) >= 10 else number_plate[-5]
    
    # Apply corrections
    part1 = correct_text(part1, "alpha")
    part2 = correct_text(part2, "numeric")
    part3 = correct_text(part3, "alpha")
    part4 = correct_text(part4, "numeric")
    
    return part1, part2, part3, part4

# Vehicle lookup functions
def get_vehicle_details_paid(plate_number):
    try:
        url = "https://rto-vehicle-information-india.p.rapidapi.com/getVehicleInfo"
        
        payload = {
            "vehicle_no": plate_number,
            "consent": "Y",
            "consent_text": "I hereby give my consent for Eccentric Labs API to fetch my information"
        }
        
        headers = {
            "x-rapidapi-key": "83ed10f183mshe1c3f0fe8025d7ap1f9c9bjsn0d1e4be443fc",
            "x-rapidapi-host": "rto-vehicle-information-india.p.rapidapi.com",
            "Content-Type": "application/json"
        }
        
        response = requests.post(url, json=payload, headers=headers)
        
        if response.status_code != 200:
            return None
        
        data = response.json()
        
        # Check if the API returned valid data
        if data.get("status") and data.get("data"):
            return data["data"]
        else:
            return None
            
    except Exception as e:
        return None

def get_vehicle_details_free(plate_number):
    try:
        url = f"https://www.carinfo.app/rto-vehicle-registration-detail/rto-details/{plate_number}"
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            return None
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract vehicle details with error handling
        try:
            make_model = soup.find('p', class_='input_vehical_layout_vehicalModel__1ABTF').text.strip()
            owner_name = soup.find('p', class_='input_vehical_layout_ownerName__NHkpi').text.strip()
            rto_number = soup.find('p', class_='expand_component_itemSubTitle__ElsYf').text.strip()
            
            # Get all subtitle elements
            subtitles = soup.find_all('p', class_='expand_component_itemSubTitle__ElsYf')
            
            # Extract details if available
            rto_address = subtitles[1].text.strip() if len(subtitles) > 1 else "Not available"
            state = subtitles[2].text.strip() if len(subtitles) > 2 else "Not available"
            phone = subtitles[3].text.strip() if len(subtitles) > 3 else "Not available"
            
            # Get website with fallback
            website = "Not available"
            if len(subtitles) > 4 and subtitles[4].find('a'):
                website = subtitles[4].find('a')['href']
            
            return {
                "maker_model": make_model,
                "owner_name": owner_name,
                "registration_no": plate_number,
                "registration_authority": rto_number,
                "rto_address": rto_address,
                "state": state,
                "rto_phone": phone,
                "website": website
            }
        except (AttributeError, IndexError) as e:
            return None
            
    except Exception as e:
        return None

# API Endpoints
@app.get("/")
async def root():
    # Redirect to the HTML frontend
    return {"message": "API is running. Access the web interface at /static/index.html"}

# Car classification endpoint
@app.post("/classify-car/")
async def classify_car(file: UploadFile = File(...)):
    global model, device
    
    if not model or not device:
        raise HTTPException(status_code=500, detail="Model not initialized")
    
    try:
        # Read image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Classify car view
        car_view = predict_car_view(image, model, device)
        
        # Convert image to base64 for response
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        return {
            "success": True,
            "car_view": car_view,
            "image_base64": img_base64
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

# License plate detection endpoint
@app.post("/detect-license-plate/")
async def detect_license(file: UploadFile = File(...)):
    global reader
    
    if not reader:
        raise HTTPException(status_code=500, detail="OCR reader not initialized")
    
    try:
        # Read image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Detect license plate
        img_base64, best_plate = detect_license_plate(image, reader)
        
        result = {
            "success": True,
            "license_detected": best_plate is not None,
            "processed_image_base64": img_base64
        }
        
        if best_plate:
            # Process the license plate
            part1, part2, part3, part4 = strict_split_number_plate(best_plate)
            
            if part1 and part2 and part3 and part4:
                corrected_plate = part1 + part2 + part3 + part4
                result["raw_plate"] = best_plate
                result["corrected_plate"] = corrected_plate
            else:
                result["raw_plate"] = best_plate
                result["corrected_plate"] = None
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

# Full processing endpoint (classification, license detection, vehicle lookup)
@app.post("/process-car-image/")
async def process_car_image(
    file: UploadFile = File(...),
    lookup_method: str = Form("free")
):
    global model, device, reader
    
    if not model or not device or not reader:
        raise HTTPException(status_code=500, detail="Models not initialized")
    
    try:
        # Read image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # 1. Classify car view
        car_view = predict_car_view(image, model, device)
        
        # 2. Detect license plate
        img_base64, best_plate = detect_license_plate(image, reader)
        
        result = {
            "success": True,
            "car_view": car_view,
            "processed_image_base64": img_base64,
            "license_detected": best_plate is not None
        }
        
        # 3. Process license plate if detected
        if best_plate:
            part1, part2, part3, part4 = strict_split_number_plate(best_plate)
            
            if part1 and part2 and part3 and part4:
                corrected_plate = part1 + part2 + part3 + part4
                result["raw_plate"] = best_plate
                result["corrected_plate"] = corrected_plate
                
                # 4. Lookup vehicle details
                if lookup_method.lower() == "paid":
                    vehicle_data = get_vehicle_details_paid(corrected_plate)
                else:
                    vehicle_data = get_vehicle_details_free(corrected_plate)
                
                if vehicle_data:
                    result["vehicle_details"] = vehicle_data
                    result["vehicle_details_found"] = True
                else:
                    result["vehicle_details_found"] = False
            else:
                result["raw_plate"] = best_plate
                result["corrected_plate"] = None
                result["vehicle_details_found"] = False
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

# Vehicle lookup endpoint
@app.post("/lookup-vehicle/")
async def lookup_vehicle(
    plate_number: str = Form(...),
    lookup_method: str = Form("free")
):
    try:
        if lookup_method.lower() == "paid":
            vehicle_data = get_vehicle_details_paid(plate_number)
        else:
            vehicle_data = get_vehicle_details_free(plate_number)
        
        if vehicle_data:
            return {
                "success": True,
                "vehicle_details": vehicle_data
            }
        else:
            return {
                "success": False,
                "message": "Vehicle details not found"
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error looking up vehicle: {str(e)}")

# Run the application
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)