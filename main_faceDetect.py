import base64
import io
import logging
import os
from datetime import datetime, timedelta
from PIL import Image
from pydantic import BaseModel
from typing import Optional, List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from skimage.metrics import structural_similarity
from sklearn.cluster import KMeans

# align face background remover
# Use rembg for background removal
from rembg import remove

import traceback

# 로깅 설정
def setup_logging():
    log_dir = "logs"
    
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # 3일 이상된 로그 파일 삭제
    cleanup_old_logs(log_dir)
    
    # 로그 파일명 설정 (현재 날짜)
    log_file = os.path.join(log_dir, f"face_detect_service_{datetime.now().strftime('%Y%m%d')}.log")
    
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)



def cleanup_old_logs(log_dir):
    now = datetime.now()
    for filename in os.listdir(log_dir):
        if filename.startswith("face_detect_service_") and filename.endswith(".log"):
            try:
                # Extract date portion from filename and parse it
                date_str = filename.split("face_detect_service_")[1].split(".")[0]  # Get YYYYMMDD part
                date_str = date_str.replace("_", "")
                file_date = datetime.strptime(date_str, "%Y%m%d")
                if (now - file_date) > timedelta(days=3):
                    os.remove(os.path.join(log_dir, filename))
                    #logger.info(f"Removed old log file: {filename}")
            except Exception as e:
                logger.error(f"Error removing old log file {filename}: {str(e)}")
                logger.error( traceback.format_exc())
                

# 전역 로거 초기화
logger = setup_logging()

                

# FastAPI 앱 생성
app = FastAPI(title="Face Detect Service", description="Face detect service with logging")

# CORS 미들웨어 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 origin 허용 (개발 환경에서만 사용)
    allow_credentials=True,
    allow_methods=["*"],  # 모든 HTTP 메소드 허용
    allow_headers=["*"],  # 모든 헤더 허용
)

class FaceDetectInput(BaseModel):
    base64image: str
    output_imgType: str

class FaceDetectOutput(BaseModel):
    base64image: str  # 입력 이미지 그대로 반환
    base64image_face: str
    base64image_faceDot: str
    base64image_faceDLib: str
    base64image_faceDLib_68_number: str # Added numbered landmarks image
    base64image_ear_detect: str  # Added ear detection image
    base64image_ear_skinColor_detect: str  # Added ear detection image
    ear_detected_count: int  # Total ear count
    ear_left_count: int  # Left ear count
    ear_left_confidence: float  # Left ear confidence
    ear_right_count: int  # Right ear count
    ear_right_confidence: float  # Right ear confidence
    resultcode: str
    resultmessage: str
    
    
    
import mediapipe as mp
import numpy as np
import cv2
import dlib

# Ear detection imports
from ear_landmark_detection import load_ear_model, detect_ears

# Initialize ear detection model
ear_model = load_ear_model()

@app.post("/detect_landmarks", response_model=FaceDetectOutput)
async def detect_face_landmarks(input_data: FaceDetectInput):
    try:
        logger.info("Processing face and ear landmarks detection")
        
        # Base64 디코딩 및 이미지 변환
        image_data = base64.b64decode(input_data.base64image)
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        image_dot = image.copy()  # dot 표시용 이미지 복사
        image_dlib = image.copy()  # dlib 결과용 이미지 복사
        image_ear = image.copy()  # ear detection용 이미지 복사
        image_ear_skinColor_mask = image.copy()  # ear detection용 이미지 복사          
        
        # Create larger image for numbered landmarks
        h, w = image.shape[:2]
        image_dlib_numbered = cv2.resize(image.copy(), (w*4, h*4))
        
        # MediaPipe Face Mesh 초기화
        mp_face_mesh = mp.solutions.face_mesh
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5
        )
        
        # DLib 초기화
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        
        # RGB 변환 (MediaPipe는 RGB 형식 사용)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create a copy of the image to ensure proper dimensions
        image_rgb_copy = image_rgb.copy()
        
        # Process the image with explicit dimensions
        mesh_results = face_mesh.process(image_rgb_copy)
        
        # DLib face detection
        gray = cv2.cvtColor(image_dlib, cv2.COLOR_BGR2GRAY)
        gray_large = cv2.cvtColor(image_dlib_numbered, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        faces_large = detector(gray_large)
        
        # Ear detection
        ear_detected_count = 0
        ear_left_count = 0
        ear_right_count = 0
        ear_left_confidence = 0.0
        ear_right_confidence = 0.0
        
        if ear_model is not None:
            try:
                ear_results = detect_ears(image_ear, ear_model, image_ear_skinColor_mask)
                ear_detected_count = ear_results['total_count']
                ear_left_count = ear_results['left_count']
                ear_right_count = ear_results['right_count']
                ear_left_confidence = ear_results['left_confidence']
                ear_right_confidence = ear_results['right_confidence']
                
                # Draw ear landmarks
                for ear in ear_results['results']:
                    for landmark in ear['landmarks']:
                        x, y = int(landmark[0]), int(landmark[1])
                        # Use different colors for left and right ears
                        color = (0, 255, 0) if ear['type'] == 'left' else (255, 0, 0)
                        cv2.circle(image_ear, (x, y), 2, color, -1)
            except Exception as e:
                logger.warning(f"Error during ear detection: {str(e)}")
                logger.error( traceback.format_exc())
        else:
            logger.warning("Ear detection model not available")
        
        if not mesh_results.multi_face_landmarks or len(faces) == 0:
            logger.warning("No face detected in the image")
            return FaceDetectOutput(
                base64image=input_data.base64image,  # 원본 이미지 반환
                base64image_face="",
                base64image_faceDot="",
                base64image_faceDLib="",
                base64image_faceDLib_68_number="",
                base64image_ear_detect="",
                base64image_ear_skinColor_detect="",
                ear_detected_count=ear_detected_count,
                ear_left_count=0,
                ear_left_confidence=0.0,
                ear_right_count=0,
                ear_right_confidence=0.0,
                resultcode="NO_FACE",
                resultmessage="No face detected in the image"
            )
        
        # 감지된 얼굴 수 계산
        face_count = len(mesh_results.multi_face_landmarks)
        
        # MediaPipe 랜드마크 그리기 (메쉬)
        for face_landmarks in mesh_results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )
            
            # 랜드마크 점으로 그리기
            for idx, landmark in enumerate(face_landmarks.landmark):
                x = int(landmark.x * image_dot.shape[1])
                y = int(landmark.y * image_dot.shape[0])
                cv2.circle(image_dot, (x, y), 2, (0, 255, 0), -1)
        
        # DLib 랜드마크 그리기
        for face in faces:
            landmarks = predictor(gray, face)
            for n in range(0, 68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                cv2.circle(image_dlib, (x, y), 2, (0, 0, 255), -1)
            
            # 얼굴 영역 표시
            cv2.rectangle(image_dlib, 
                        (face.left(), face.top()),
                        (face.right(), face.bottom()),
                        (0, 255, 0), 2)
                        
        # Draw numbered landmarks on large image
        for face in faces_large:
            landmarks = predictor(gray_large, face)
            for n in range(0, 68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                # Draw larger circle
                cv2.circle(image_dlib_numbered, (x, y), 6, (0, 0, 255), -1)
                # Add number label
                cv2.putText(image_dlib_numbered, str(n), (x+10, y+10),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Draw face rectangle
            cv2.rectangle(image_dlib_numbered,
                        (face.left(), face.top()),
                        (face.right(), face.bottom()),
                        (0, 255, 0), 4)
            
        # 이미지를 base64로 인코딩 (메쉬)
        _, buffer = cv2.imencode(f'.{input_data.output_imgType}', image)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # dot 이미지를 base64로 인코딩
        _, buffer_dot = cv2.imencode(f'.{input_data.output_imgType}', image_dot)
        img_dot_base64 = base64.b64encode(buffer_dot).decode('utf-8')
        
        # dlib 이미지를 base64로 인코딩
        _, buffer_dlib = cv2.imencode(f'.{input_data.output_imgType}', image_dlib)
        img_dlib_base64 = base64.b64encode(buffer_dlib).decode('utf-8')
        
        # numbered dlib 이미지를 base64로 인코딩
        _, buffer_dlib_numbered = cv2.imencode(f'.{input_data.output_imgType}', image_dlib_numbered)
        img_dlib_numbered_base64 = base64.b64encode(buffer_dlib_numbered).decode('utf-8')
        
        # ear detection 이미지를 base64로 인코딩
        _, buffer_ear = cv2.imencode(f'.{input_data.output_imgType}', image_ear)
        img_ear_base64 = base64.b64encode(buffer_ear).decode('utf-8')
        
        # ear skin color detection 이미지를 base64로 인코딩
        _, buffer_ear_skinColor = cv2.imencode(f'.{input_data.output_imgType}', image_ear_skinColor_mask)
        img_ear_skinColor_base64 = base64.b64encode(buffer_ear_skinColor).decode('utf-8')
        
        logger.info("Successfully detected face and ear landmarks")
        
        return FaceDetectOutput(
            base64image=input_data.base64image,  # 원본 이미지 반환
            base64image_face=img_base64,  # base64로 인코딩된 메쉬 이미지
            base64image_faceDot=img_dot_base64,  # base64로 인코딩된 dot 이미지
            base64image_faceDLib=img_dlib_base64,  # base64로 인코딩된 dlib 이미지
            base64image_faceDLib_68_number=img_dlib_numbered_base64,  # base64로 인코딩된 numbered dlib 이미지
            base64image_ear_detect=img_ear_base64,  # base64로 인코딩된 ear detection 이미지
            base64image_ear_skinColor_detect=img_ear_skinColor_base64,  # base64로 인코딩된 ear skin color detection 이미지
            ear_detected_count=ear_detected_count,  # 감지된 귀의 총 수
            ear_left_count=ear_left_count,  # 왼쪽 귀 감지 수
            ear_left_confidence=round(float(ear_left_confidence), 2),  # 왼쪽 귀 신뢰도
            ear_right_count=ear_right_count,  # 오른쪽 귀 감지 수
            ear_right_confidence=round(float(ear_right_confidence), 2),  # 오른쪽 귀 신뢰도
            resultcode="SUCCESS",
            resultmessage=f"Detected {face_count} faces and {ear_detected_count} ears"
        )
        
    except Exception as e:
        logger.error(f"Error processing face and ear landmarks: {str(e)}")
        logger.error( traceback.format_exc())
        return FaceDetectOutput(
            base64image=input_data.base64image,  # 원본 이미지 반환
            base64image_face="",
            base64image_faceDot="",
            base64image_faceDLib="",
            base64image_faceDLib_68_number="",
            base64image_ear_detect="",
            base64image_ear_skinColor_detect="",
            ear_detected_count=0,
            ear_left_count=0,
            ear_left_confidence=0.0,
            ear_right_count=0,
            ear_right_confidence=0.0,
            resultcode="ERROR",
            resultmessage=str(e)
        )

class FaceAlignInput(BaseModel):
    base64image: str
    output_imgType: str

class FaceAlignOutput(BaseModel):
    base64image_align: str
    base64image_bg_mask: str  # Added background mask image
    base64image_bg_mask_green: str  # Added green background mask image
    background_color: list[int]
    resultcode: str
    resultmessage: str

@app.post("/align_face", response_model=FaceAlignOutput)
async def align_face(input_data: FaceAlignInput):
    try:
        logger.info(f"Processing face alignment for file: {input_data.output_imgType}")
        
        # Base64 디코딩 및 이미지 변환
        image_data = base64.b64decode(input_data.base64image)
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # MediaPipe Face Mesh 초기화
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5
        )

        # RGB 변환
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create a copy of the image to ensure proper dimensions
        image_rgb_copy = image_rgb.copy()
        
        # Process the image with explicit dimensions
        mesh_results = face_mesh.process(image_rgb_copy)

        if not mesh_results.multi_face_landmarks:
            logger.warning("No face detected in the image")
            return FaceAlignOutput(
                base64image_align="",
                base64image_bg_mask="",
                base64image_bg_mask_green="",
                background_color=[],
                resultcode="NO_FACE",
                resultmessage="No face detected in the image"
            )

        # 눈 랜드마크 인덱스 (MediaPipe Face Mesh)
        LEFT_EYE = [33]  # 왼쪽 눈 중심점
        RIGHT_EYE = [263]  # 오른쪽 눈 중심점

        # 눈 좌표 추출
        face_landmarks = mesh_results.multi_face_landmarks[0]
        left_eye = np.array([face_landmarks.landmark[LEFT_EYE[0]].x * image.shape[1],
                            face_landmarks.landmark[LEFT_EYE[0]].y * image.shape[0]])
        right_eye = np.array([face_landmarks.landmark[RIGHT_EYE[0]].x * image.shape[1],
                             face_landmarks.landmark[RIGHT_EYE[0]].y * image.shape[0]])

        # 두 눈의 각도 계산
        angle = np.degrees(np.arctan2(right_eye[1] - left_eye[1],
                                    right_eye[0] - left_eye[0]))

        # 이미지 중심점
        center = (image.shape[1] // 2, image.shape[0] // 2)

        # 회전 매트릭스 계산
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Calculate new image dimensions to minimize padding
        cos = np.abs(rotation_matrix[0, 0])
        sin = np.abs(rotation_matrix[0, 1])
        
        new_w = int((image.shape[0] * sin) + (image.shape[1] * cos))
        new_h = int((image.shape[0] * cos) + (image.shape[1] * sin))
        
        # Adjust the matrix translation
        rotation_matrix[0, 2] += (new_w / 2) - center[0]
        rotation_matrix[1, 2] += (new_h / 2) - center[1]
        
        # Convert cv2 image to PIL Image for rembg
        image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Remove background using rembg
        output_pil = remove(image_pil)
        
        # Convert back to cv2 format
        bg_mask = cv2.cvtColor(np.array(output_pil), cv2.COLOR_RGBA2BGR)
        
        # Create white background
        white_bg = np.ones_like(image) * 255
        # Create green background
        green_bg = np.ones_like(image) * [0, 255, 0]  # BGR format for green
        
        alpha = output_pil.split()[-1]
        alpha_cv = cv2.cvtColor(np.array(alpha), cv2.COLOR_GRAY2BGR) / 255.0
        
        bg_mask = (alpha_cv * image + (1 - alpha_cv) * white_bg).astype(np.uint8)
        bg_mask_green = (alpha_cv * image + (1 - alpha_cv) * green_bg).astype(np.uint8)
        
        # Rotate all images
        rotated_image = cv2.warpAffine(image, rotation_matrix, (new_w, new_h), borderValue=(255,255,255))
        rotated_bg_mask = cv2.warpAffine(bg_mask, rotation_matrix, (new_w, new_h), borderValue=(255,255,255))
        rotated_bg_mask_green = cv2.warpAffine(bg_mask_green, rotation_matrix, (new_w, new_h), borderValue=(0,255,0))

        # Resize back to original dimensions if needed
        if new_w != image.shape[1] or new_h != image.shape[0]:
            rotated_image = cv2.resize(rotated_image, (image.shape[1], image.shape[0]))
            rotated_bg_mask = cv2.resize(rotated_bg_mask, (image.shape[1], image.shape[0]))
            rotated_bg_mask_green = cv2.resize(rotated_bg_mask_green, (image.shape[1], image.shape[0]))

        # Encode images to base64
        _, buffer = cv2.imencode(f'.{input_data.output_imgType}', rotated_image)
        rotated_image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        _, buffer_mask = cv2.imencode(f'.{input_data.output_imgType}', rotated_bg_mask)
        bg_mask_base64 = base64.b64encode(buffer_mask).decode('utf-8')

        _, buffer_mask_green = cv2.imencode(f'.{input_data.output_imgType}', rotated_bg_mask_green)
        bg_mask_green_base64 = base64.b64encode(buffer_mask_green).decode('utf-8')

        logger.info(f"Successfully aligned face for file: {input_data.output_imgType}")

        return FaceAlignOutput(
            base64image_align=rotated_image_base64,
            base64image_bg_mask=bg_mask_base64,
            base64image_bg_mask_green=bg_mask_green_base64,
            background_color=[255, 255, 255],  # White background color
            resultcode="SUCCESS",
            resultmessage=f"Face aligned successfully. Rotation angle: {angle:.2f} degrees"
        )

    except Exception as e:
        logger.error(f"Error processing face alignment for file {input_data.output_imgType}: {str(e)}")
        logger.error( traceback.format_exc())
        return FaceAlignOutput(
            base64image_align="",
            base64image_bg_mask="",
            base64image_bg_mask_green="",
            background_color=[],
            resultcode="ERROR",
            resultmessage=str(e)
        )

# DeepFace를 사용하여 얼굴 유사도 비교
from deepface import DeepFace

class FaceSimilarityInput(BaseModel):
    base64image1: str
    base64image2: str
    filename1: str 
    filename2: str

class FaceSimilarityOutput(BaseModel):
    similarity_score: float
    resultcode: str
    resultmessage: str

@app.post("/face-similarity", response_model=FaceSimilarityOutput)
async def compare_faces(input_data: FaceSimilarityInput):
    try:
        logger.info(f"Processing face similarity request for files: {input_data.filename1} and {input_data.filename2}")

        # Base64 디코딩 및 이미지 변환 - 첫 번째 이미지
        image_data1 = base64.b64decode(input_data.base64image1)
        nparr1 = np.frombuffer(image_data1, np.uint8)
        image1 = cv2.imdecode(nparr1, cv2.IMREAD_COLOR)

        # Base64 디코딩 및 이미지 변환 - 두 번째 이미지
        image_data2 = base64.b64decode(input_data.base64image2)
        nparr2 = np.frombuffer(image_data2, np.uint8)
        image2 = cv2.imdecode(nparr2, cv2.IMREAD_COLOR)

        # BGR to RGB 변환
        image1_rgb = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        image2_rgb = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

        
        result = DeepFace.verify(
            img1_path=image1_rgb,
            img2_path=image2_rgb,
            model_name="Facenet",
            enforce_detection=True,
            detector_backend="mtcnn"
        )

        # 유사도 점수 추출 (0-100 범위로 변환)
        similarity_score = (1 - result["distance"]) * 100
        similarity_score = max(0, min(100, similarity_score))  # 0-100 범위로 제한

        logger.info(f"Successfully compared faces using Facenet. Similarity score: {similarity_score:.2f}")

        return FaceSimilarityOutput(
            similarity_score=similarity_score,
            resultcode="SUCCESS",
            resultmessage=f"Face similarity comparison completed using Facenet. Score: {similarity_score:.2f}"
        )

    except Exception as e:
        logger.error(f"Error processing face similarity: {str(e)}")
        logger.error( traceback.format_exc())
        return FaceSimilarityOutput(
            similarity_score=0.0,
            resultcode="ERROR",
            resultmessage=str(e)
        )














class PassportPhotoQualityInput(BaseModel):
    base64image: str
    output_imgType: str

class PassportPhotoQualityOutput(BaseModel):
    base64image: str  # 입력 이미지 그대로 반환
    base64image_face_detect: str  # Face detection visualization
    base64image_head_detect: str  # Head detection visualization
    base64image_bg_removed: str  # Background removed image
    base64image_bg_removed_canny: str  # Background removed image with canny edges
    base64image_bg_removed_canny_upd_crown: str  # Background removed image with updated crown point
    base64image_ear_detect: str  # Added ear detection image
    base64image_ear_skinColor_detect: str  # Added ear detection image
    ear_detected_count: int  # Total ear count
    ear_left_count: int  # Left ear count
    ear_left_confidence: float  # Left ear confidence
    ear_right_count: int  # Right ear count
    ear_right_confidence: float  # Right ear confidence
    face_height_proportion: float  # Face height / image height ratio
    face_width_proportion: float  # Face width / image width ratio
    head_height_proportion: float  # Head height / image height ratio
    head_width_proportion: float  # Head width / image width ratio
    # Image Quality Metrics
    blur_score: float  # Laplacian-based blur detection
    pixelation_score: float  # Pixelation detection
    white_noise_score: float  # White noise estimation
    contrast_score: float  # Contrast measurement
    general_illumination_score: float  # Overall brightness
    # Face Quality Metrics
    face_position_score: float  # Face position in image
    face_pose_score: float  # Roll, pitch, yaw estimation
    expression_score: float  # Facial expression
    eyes_open_score: float  # Eyes open/closed
    eyes_direction_score: float  # Eye direction
    mouth_open_score: float  # Mouth open/closed
    # Occlusion Detection
    hair_over_face_score: float  # Hair over face
    sunglasses_score: float  # Sunglasses detection
    glasses_reflection_score: float  # Light reflections on glasses
    glasses_frame_score: float  # Wide frames of glasses
    glasses_covering_score: float  # Frames covering eyes
    hat_score: float  # Hat detection
    veil_score: float  # Veil detection
    # Color and Lighting
    skin_color_score: float  # Natural skin color
    red_eyes_score: float  # Red eye detection
    skin_reflection_score: float  # Light reflections on skin
    shadow_face_score: float  # Shadows over face
    shadow_background_score: float  # Shadows in background
    # Background Quality
    background_uniformity_score: float  # Background uniformity
    ink_marks_score: float  # Presence of ink marks
    other_faces_score: float  # Detection of other faces
    # Overall Score
    overall_score: float  # Weighted average of all scores
    resultcode: str
    resultmessage: str

@app.post("/check_pspt_photo_quality", response_model=PassportPhotoQualityOutput)
async def check_passport_photo_quality(input_data: PassportPhotoQualityInput):
    try:
        logger.info("Checking passport photo quality compliance")
        
        # Validate input data
        if not input_data.base64image or not input_data.output_imgType:
            raise ValueError("Missing required input parameters")
            
        # Validate base64 image
        try:
            image_data = base64.b64decode(input_data.base64image)
            if not image_data:
                raise ValueError("Invalid base64 image data")
        except Exception as e:
            raise ValueError(f"Invalid base64 image format: {str(e)}")
            
        # Validate image type
        valid_image_types = ['jpg', 'jpeg', 'png']
        if input_data.output_imgType.lower() not in valid_image_types:
            raise ValueError(f"Invalid image type. Supported types: {', '.join(valid_image_types)}")
        
        # Base64 디코딩 및 이미지 변환
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Failed to decode image")
            
        h, w = image.shape[:2]

        # Ear detection
        image_ear = image.copy()
        ear_detected_count = 0
        ear_left_count = 0
        ear_right_count = 0
        ear_left_confidence = 0.0
        ear_right_confidence = 0.0
        
        image_ear_skinColor_mask = image.copy()
        
        if ear_model is not None:
            try:
                ear_results = detect_ears(image_ear, ear_model, image_ear_skinColor_mask)
                ear_detected_count = ear_results['total_count']
                ear_left_count = ear_results['left_count']
                ear_right_count = ear_results['right_count']
                ear_left_confidence = ear_results['left_confidence']
                ear_right_confidence = ear_results['right_confidence']
                
                # Draw ear landmarks
                for ear in ear_results['results']:
                    for landmark in ear['landmarks']:
                        x, y = int(landmark[0]), int(landmark[1])
                        # Use different colors for left and right ears
                        color = (0, 255, 0) if ear['type'] == 'left' else (255, 0, 0)
                        cv2.circle(image_ear, (x, y), 2, color, -1)
            except Exception as e:
                logger.warning(f"Error during ear detection: {str(e)}")
        else:
            logger.warning("Ear detection model not available")
        
        # Encode ear detection image to base64
        _, buffer_ear = cv2.imencode(f'.{input_data.output_imgType}', image_ear)
        base64image_ear_detect = base64.b64encode(buffer_ear).decode('utf-8')
        
        # Encode ear skin color detection image to base64
        _, buffer_ear_skinColor = cv2.imencode(f'.{input_data.output_imgType}', image_ear_skinColor_mask)
        base64image_ear_skinColor_detect = base64.b64encode(buffer_ear_skinColor).decode('utf-8')

        # Calculate blur score using Laplacian variance
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        blur_score = min(100, max(0, blur_score))  # Normalize to 0-100

        # Calculate pixelation score
        pixelation_score = 100 * (1 - np.mean(np.abs(np.diff(gray, axis=0))) / 255)

        # Calculate white noise score
        noise = np.std(gray) / np.mean(gray)
        white_noise_score = 100 * (1 - min(1, noise))

        # Calculate contrast score
        contrast = np.std(gray) / 255
        contrast_score = 100 * contrast

        # Calculate general illumination score
        mean_brightness = np.mean(gray) / 255
        general_illumination_score = 100 * (1 - abs(0.5 - mean_brightness))

        # Remove background using rembg
        image_no_bg = remove(image)
        
        # Convert background removed image to base64
        _, buffer = cv2.imencode(f'.{input_data.output_imgType}', image_no_bg)
        base64image_bg_removed = base64.b64encode(buffer).decode('utf-8')
        
        # Create canny edge image from background removed image
        gray = cv2.cvtColor(image_no_bg, cv2.COLOR_BGR2GRAY)
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # Apply morphological closing
        kernel = np.ones((5,5), np.uint8)
        
                
        # Apply erosion or opening
        eroded = cv2.erode(blurred, kernel, iterations=1)        
        closed = cv2.morphologyEx(blurred, cv2.MORPH_OPEN, kernel)

        
        # Apply Canny edge detection
        edges = cv2.Canny(eroded, 50, 150)
        #edges = cv2.Canny(closed, 50, 150)
        
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create copy of background removed image for drawing
        image_with_edges = image_no_bg.copy()
        
        # Draw contours with magenta color and 5px thickness
        cv2.drawContours(image_with_edges, contours, -1, (255, 0, 255), 5)
        
        # Convert edge image to base64
        _, buffer = cv2.imencode(f'.{input_data.output_imgType}', image_with_edges)
        base64image_bg_removed_canny = base64.b64encode(buffer).decode('utf-8')
        
        # MediaPipe Face Mesh 초기화
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5,
            refine_landmarks=True
        )
        
        # RGB 변환
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb_copy = image_rgb.copy()
        mesh_results = face_mesh.process(image_rgb_copy)
        
        if not mesh_results.multi_face_landmarks:
            return PassportPhotoQualityOutput(
                base64image=input_data.base64image,
                base64image_face_detect="",
                base64image_head_detect="",
                base64image_bg_removed=base64image_bg_removed,
                base64image_bg_removed_canny=base64image_bg_removed_canny,
                base64image_bg_removed_canny_upd_crown="",
                base64image_ear_detect="",
                base64image_ear_skinColor_detect="",
                ear_detected_count=0,
                ear_left_count=0,
                ear_left_confidence=0.0,
                ear_right_count=0,
                ear_right_confidence=0.0,
                face_height_proportion=0.0,
                face_width_proportion=0.0,
                head_height_proportion=0.0,
                head_width_proportion=0.0,
                blur_score=blur_score,
                pixelation_score=pixelation_score,
                white_noise_score=white_noise_score,
                contrast_score=contrast_score,
                general_illumination_score=general_illumination_score,
                face_position_score=0.0,
                face_pose_score=0.0,
                expression_score=0.0,
                eyes_open_score=0.0,
                eyes_direction_score=0.0,
                mouth_open_score=0.0,
                hair_over_face_score=0.0,
                sunglasses_score=0.0,
                glasses_reflection_score=0.0,
                glasses_frame_score=0.0,
                glasses_covering_score=0.0,
                hat_score=0.0,
                veil_score=0.0,
                skin_color_score=0.0,
                red_eyes_score=0.0,
                skin_reflection_score=0.0,
                shadow_face_score=0.0,
                shadow_background_score=0.0,
                background_uniformity_score=0.0,
                ink_marks_score=0.0,
                other_faces_score=0.0,
                overall_score=0.0,
                resultcode="NO_FACE",
                resultmessage="No face detected in image"
            )
        
        landmarks = mesh_results.multi_face_landmarks[0].landmark
        
        # Get head measurements
        crown_point = landmarks[10]  # Top of head
        chin_point = landmarks[152]  # Bottom of chin
        left_ear = landmarks[234]  # Left ear
        right_ear = landmarks[454]  # Right ear
        
        
        # Calculate proportions Face Size
        face_height = abs(crown_point.y - chin_point.y) * h
        face_width = abs(right_ear.x - left_ear.x) * w
        
        face_height_proportion = face_height / h
        face_width_proportion = face_width / w
        
                # Create detection visualization with updated crown point
        image_with_face_points = image_no_bg.copy()
        
        # Draw red dots at key points including updated crown point
        points = [
            (int(crown_point.x * w), int(crown_point.y * h)),
            (int(chin_point.x * w), int(chin_point.y * h)),
            (int(left_ear.x * w), int(left_ear.y * h)),
            (int(right_ear.x * w), int(right_ear.y * h))
        ]
        
        for point in points:
            cv2.circle(image_with_face_points, point, 5, (0,0,255), -1)
         
 
        # Convert visualization to base64
        _, buffer = cv2.imencode(f'.{input_data.output_imgType}', image_with_face_points)
        base64image_face_detect = base64.b64encode(buffer).decode('utf-8')
        
        
        
        
        
        
        
        

        # Find min Y from contours
        #min_contour_y = 0
        min_contour_y = image.shape[1] / 2
        
        for contour in contours:
            min_y = np.min(contour[:, :, 1])
            min_contour_y = min(min_contour_y, min_y)
        min_contour_y = min_contour_y / h  # Normalize to 0-1 range

        # Update crown point Y if contour Y is higher
        if min_contour_y < crown_point.y:  # Note: Y coordinates are inverted (0 is top)
            crown_point.y = min_contour_y
        
        # Calculate proportions Head Size
        head_height = abs(crown_point.y - chin_point.y) * h
        head_width = abs(right_ear.x - left_ear.x) * w
        
        head_height_proportion = head_height / h
        head_width_proportion = head_width / w
        
        # Create detection visualization with updated crown point
        image_with_updated_points = image_no_bg.copy()
        
        # Draw red dots at key points including updated crown point
        points = [
            (int(crown_point.x * w), int(crown_point.y * h)),
            (int(chin_point.x * w), int(chin_point.y * h)),
            (int(left_ear.x * w), int(left_ear.y * h)),
            (int(right_ear.x * w), int(right_ear.y * h))
        ]
        
        for point in points:
            cv2.circle(image_with_updated_points, point, 5, (0,0,255), -1)
            
        # Convert updated points visualization to base64
        _, buffer = cv2.imencode(f'.{input_data.output_imgType}', image_with_updated_points)
        base64image_bg_removed_canny_upd_crown = base64.b64encode(buffer).decode('utf-8')
        
        # Create detection visualization
        image_with_dots = image.copy()
        
        # Draw red dots at key points
        points = [
            (int(crown_point.x * w), int(crown_point.y * h)),
            (int(chin_point.x * w), int(chin_point.y * h)),
            (int(left_ear.x * w), int(left_ear.y * h)),
            (int(right_ear.x * w), int(right_ear.y * h))
        ]
        
        for point in points:
            cv2.circle(image_with_dots, point, 5, (0,0,255), -1)
            
        # Convert visualization to base64
        _, buffer = cv2.imencode(f'.{input_data.output_imgType}', image_with_dots)
        base64image_head_detect = base64.b64encode(buffer).decode('utf-8')
        
        # 2. Face Quality Metrics
        # Face position
        face_center_x = np.mean([landmarks[i].x for i in [1, 152, 454]])
        face_center_y = np.mean([landmarks[i].y for i in [1, 152, 454]])
        face_position_score = 100 * (1 - (abs(0.5 - face_center_x) + abs(0.5 - face_center_y)))
        
        # Face pose (roll, pitch, yaw)
        nose_tip = landmarks[4]
        left_eye = landmarks[33]
        right_eye = landmarks[263]
        face_pose_score = 100 * (1 - abs(left_eye.z - right_eye.z))
        
        # Expression
        mouth_height = abs(landmarks[13].y - landmarks[14].y)
        mouth_width = abs(landmarks[78].x - landmarks[308].x)
        expression_score = 100 * (1 - mouth_height * 2) * (1 - abs(mouth_width - 0.5) * 2)
        
        # Eyes open/closed
        left_eye_height = abs(landmarks[159].y - landmarks[145].y)
        right_eye_height = abs(landmarks[386].y - landmarks[374].y)
        eyes_open_score = 100 * (left_eye_height + right_eye_height) / 2
        
        # Eyes direction
        left_pupil = landmarks[468]
        right_pupil = landmarks[473]
        eyes_direction_score = 100 * (1 - (abs(left_pupil.x - left_eye.x) + abs(right_pupil.x - right_eye.x)))
        
        # Mouth open/closed
        mouth_open_score = 100 * (1 - abs(landmarks[13].y - landmarks[14].y))
        
        # 3. Occlusion Detection
        # Hair over face (using face mesh coverage)
        hair_over_face_score = 100 * len([l for l in landmarks if l.visibility > 0.9]) / len(landmarks)
        
        # Sunglasses detection
        eye_region = image[int(landmarks[159].y*h):int(landmarks[386].y*h),
                         int(landmarks[33].x*w):int(landmarks[263].x*w)]
        sunglasses_score = 100 * (1 - np.mean(eye_region) / 255.0)
        
        # Glasses reflection
        glasses_reflection_score = 100 * (1 - np.std(eye_region) / 128.0)
        
        # Glasses frame
        glasses_frame_score = 100 * (1 - np.std(eye_region) / 128.0)
        
        # Glasses covering eyes
        glasses_covering_score = 100 * (1 - np.mean(eye_region) / 255.0)
        
        # Hat detection
        forehead_region = image[int(landmarks[10].y*h):int(landmarks[152].y*h),
                              int(landmarks[234].x*w):int(landmarks[454].x*w)]
        hat_score = 100 * (1 - np.std(forehead_region) / 128.0)
        
        # Veil detection
        lower_face_region = image[int(landmarks[152].y*h):int(landmarks[152].y*h+h/4),
                                int(landmarks[234].x*w):int(landmarks[454].x*w)]
        veil_score = 100 * (1 - np.std(lower_face_region) / 128.0)
        
        # 4. Color and Lighting
        # Skin color
        face_roi = image[int(face_center_y*h-h/4):int(face_center_y*h+h/4),
                        int(face_center_x*w-w/4):int(face_center_x*w+w/4)]
        skin_color_score = 100 * (1 - abs(np.mean(face_roi) - 128) / 128)
        
        # Red eyes
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        red_mask = cv2.inRange(hsv, (0,50,50), (10,255,255))
        red_eyes_score = 100 * (1 - np.sum(red_mask) / (h * w * 255))
        
        # Skin reflection
        skin_reflection_score = 100 * (1 - np.std(face_roi) / 128.0)
        
        # Shadows over face
        shadow_face_score = 100 * (1 - np.std(face_roi) / 128.0)
        
        # Shadows in background
        background_region = image[0:int(h/4), 0:w]
        shadow_background_score = 100 * (1 - np.std(background_region) / 128.0)
        
        # 5. Background Quality
        # Background uniformity
        background_uniformity_score = 100 * (1 - np.std(background_region) / 128.0)
        
        # Ink marks
        ink_marks_score = 100 * (1 - np.std(background_region) / 128.0)
        
        # Other faces
        other_faces_score = 100 * (1 - len(mesh_results.multi_face_landmarks) / 2)
        
        # Calculate overall score with weights
        weights = {
            'blur': 0.05,
            'pixelation': 0.05,
            'white_noise': 0.05,
            'contrast': 0.05,
            'general_illumination': 0.05,
            'face_position': 0.05,
            'face_pose': 0.05,
            'expression': 0.05,
            'eyes_open': 0.05,
            'eyes_direction': 0.05,
            'mouth_open': 0.05,
            'hair_over_face': 0.05,
            'sunglasses': 0.05,
            'glasses_reflection': 0.05,
            'glasses_frame': 0.05,
            'glasses_covering': 0.05,
            'hat': 0.05,
            'veil': 0.05,
            'skin_color': 0.05,
            'red_eyes': 0.05,
            'skin_reflection': 0.05,
            'shadow_face': 0.05,
            'shadow_background': 0.05,
            'background_uniformity': 0.05,
            'ink_marks': 0.05,
            'other_faces': 0.05
        }
        
        scores = {
            'blur': blur_score,
            'pixelation': pixelation_score,
            'white_noise': white_noise_score,
            'contrast': contrast_score,
            'general_illumination': general_illumination_score,
            'face_position': face_position_score,
            'face_pose': face_pose_score,
            'expression': expression_score,
            'eyes_open': eyes_open_score,
            'eyes_direction': eyes_direction_score,
            'mouth_open': mouth_open_score,
            'hair_over_face': hair_over_face_score,
            'sunglasses': sunglasses_score,
            'glasses_reflection': glasses_reflection_score,
            'glasses_frame': glasses_frame_score,
            'glasses_covering': glasses_covering_score,
            'hat': hat_score,
            'veil': veil_score,
            'skin_color': skin_color_score,
            'red_eyes': red_eyes_score,
            'skin_reflection': skin_reflection_score,
            'shadow_face': shadow_face_score,
            'shadow_background': shadow_background_score,
            'background_uniformity': background_uniformity_score,
            'ink_marks': ink_marks_score,
            'other_faces': other_faces_score
        }
        
        # Replace nan values with 0 and ensure all scores are within valid range
        for key in scores:
            if np.isnan(scores[key]) or not np.isfinite(scores[key]):
                scores[key] = 0.0
            scores[key] = max(0.0, min(100.0, scores[key]))  # Ensure score is between 0 and 100
        
        overall_score = sum(score * weights[key] for key, score in scores.items())
        overall_score = max(0.0, min(100.0, overall_score))  # Ensure overall score is between 0 and 100
        
        return PassportPhotoQualityOutput(
            base64image=input_data.base64image,
            base64image_face_detect=base64image_face_detect,
            base64image_head_detect=base64image_head_detect,
            base64image_bg_removed=base64image_bg_removed,
            base64image_bg_removed_canny=base64image_bg_removed_canny,
            base64image_bg_removed_canny_upd_crown=base64image_bg_removed_canny_upd_crown,
            base64image_ear_detect=base64image_ear_detect,  
            base64image_ear_skinColor_detect=base64image_ear_skinColor_detect,
            ear_detected_count=ear_detected_count,
            ear_left_count=ear_left_count,
            ear_left_confidence=round(float(ear_left_confidence), 2),
            ear_right_count=ear_right_count,
            ear_right_confidence=round(float(ear_right_confidence), 2),
            face_height_proportion=round(float(face_height_proportion), 4),
            face_width_proportion=round(float(face_width_proportion), 4),
            head_height_proportion=round(float(head_height_proportion), 4),
            head_width_proportion=round(float(head_width_proportion), 4),
            blur_score=round(float(scores['blur']), 2),
            pixelation_score=round(float(scores['pixelation']), 2),
            white_noise_score=round(float(scores['white_noise']), 2),
            contrast_score=round(float(scores['contrast']), 2),
            general_illumination_score=round(float(scores['general_illumination']), 2),
            face_position_score=round(float(scores['face_position']), 2),
            face_pose_score=round(float(scores['face_pose']), 2),
            expression_score=round(float(scores['expression']), 2),
            eyes_open_score=round(float(scores['eyes_open']), 2),
            eyes_direction_score=round(float(scores['eyes_direction']), 2),
            mouth_open_score=round(float(scores['mouth_open']), 2),
            hair_over_face_score=round(float(scores['hair_over_face']), 2),
            sunglasses_score=round(float(scores['sunglasses']), 2),
            glasses_reflection_score=round(float(scores['glasses_reflection']), 2),
            glasses_frame_score=round(float(scores['glasses_frame']), 2),
            glasses_covering_score=round(float(scores['glasses_covering']), 2),
            hat_score=round(float(scores['hat']), 2),
            veil_score=round(float(scores['veil']), 2),
            skin_color_score=round(float(scores['skin_color']), 2),
            red_eyes_score=round(float(scores['red_eyes']), 2),
            skin_reflection_score=round(float(scores['skin_reflection']), 2),
            shadow_face_score=round(float(scores['shadow_face']), 2),
            shadow_background_score=round(float(scores['shadow_background']), 2),
            background_uniformity_score=round(float(scores['background_uniformity']), 2),
            ink_marks_score=round(float(scores['ink_marks']), 2),
            other_faces_score=round(float(scores['other_faces']), 2),
            overall_score=round(float(overall_score), 2),
            resultcode="SUCCESS",
            resultmessage="Passport photo quality check completed successfully"
        )
        
    except Exception as e:
        logger.error(f"Error checking passport photo quality: {str(e)}")
        return PassportPhotoQualityOutput(
            base64image=input_data.base64image,
            base64image_face_detect="",
            base64image_head_detect="",
            base64image_bg_removed="",
            base64image_bg_removed_canny="",
            base64image_bg_removed_canny_upd_crown="",
            base64image_ear_detect="",
            base64image_ear_skinColor_detect="",
            ear_detected_count=0,
            ear_left_count=0,
            ear_left_confidence=0.0,
            ear_right_count=0,
            ear_right_confidence=0.0,
            face_height_proportion=0.0,
            face_width_proportion=0.0,
            head_height_proportion=0.0,
            head_width_proportion=0.0,
            blur_score=0.0,
            pixelation_score=0.0,
            white_noise_score=0.0,
            contrast_score=0.0,
            general_illumination_score=0.0,
            face_position_score=0.0,
            face_pose_score=0.0,
            expression_score=0.0,
            eyes_open_score=0.0,
            eyes_direction_score=0.0,
            mouth_open_score=0.0,
            hair_over_face_score=0.0,
            sunglasses_score=0.0,
            glasses_reflection_score=0.0,
            glasses_frame_score=0.0,
            glasses_covering_score=0.0,
            hat_score=0.0,
            veil_score=0.0,
            skin_color_score=0.0,
            red_eyes_score=0.0,
            skin_reflection_score=0.0,
            shadow_face_score=0.0,
            shadow_background_score=0.0,
            background_uniformity_score=0.0,
            ink_marks_score=0.0,
            other_faces_score=0.0,
            overall_score=0.0,
            resultcode="ERROR",
            resultmessage=str(e)
        )



















# import os
# from dotenv import load_dotenv

# # Load environment variables
# load_dotenv()

# # Get threshold values from environment variables with defaults
# FACE_SIZE_THRESHOLD = float(os.getenv('FACE_SIZE_THRESHOLD', '0.3'))  # Face should take 70-80% of image height
# EXPRESSION_THRESHOLD = float(os.getenv('EXPRESSION_THRESHOLD', '0.7'))  # Neutral expression score threshold
# BACKGROUND_UNIFORMITY_THRESHOLD = float(os.getenv('BACKGROUND_UNIFORMITY_THRESHOLD', '0.9'))  # Background uniformity threshold
# OVERALL_PASS_THRESHOLD = float(os.getenv('OVERALL_PASS_THRESHOLD', '0.8'))  # Overall passing score threshold

# class ICAOCheckOutput(BaseModel):
#     base64image: str  # 입력 이미지 그대로 반환
#     face_size_score: float
#     expression_score: float 
#     background_score: float
#     overall_score: float
#     face_size_pass: bool
#     expression_pass: bool
#     background_pass: bool
#     overall_pass: bool
#     # New scores with default values
#     face_position_score: float
#     face_pose_score: float
#     face_occlusion_score: float
#     lips_open_score: float
#     eyes_open_score: float
#     eye_color_score: float
#     eye_direction_score: float
#     shadow_score: float
#     exposure_score: float
#     sharpness_score: float
#     shoulder_score: float
#     # New pass/fail indicators with default values
#     face_position_pass: bool
#     face_pose_pass: bool
#     face_occlusion_pass: bool
#     lips_open_pass: bool
#     eyes_open_pass: bool
#     eye_color_pass: bool
#     eye_direction_pass: bool
#     shadow_pass: bool
#     exposure_pass: bool
#     sharpness_pass: bool
#     shoulder_pass: bool
#     resultcode: str
#     resultmessage: str



    

if __name__ == "__main__":
    logger.info("Starting Face Detect Service")
    uvicorn.run(app, host="0.0.0.0", port=8801)








# @app.post("/check_passport_photo", response_model=ICAOCheckOutput)
# async def check_passport_photo(input_data: FaceDetectInput):
#     try:
#         logger.info("Checking passport photo compliance")
        
#         # Decode base64 image
#         image_data = base64.b64decode(input_data.base64image)
#         nparr = np.frombuffer(image_data, np.uint8)
#         image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#         h, w = image.shape[:2]
        
#         # Initialize MediaPipe Face Mesh
#         mp_face_mesh = mp.solutions.face_mesh
#         face_mesh = mp_face_mesh.FaceMesh(
#             static_image_mode=True,
#             max_num_faces=1,
#             min_detection_confidence=0.5
#         )
        
#         # Convert to RGB
#         image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         mesh_results = face_mesh.process(image_rgb)
        
#         if not mesh_results.multi_face_landmarks:
#             return ICAOCheckOutput(
#                 base64image=input_data.base64image,
#                 face_size_score=0.0,
#                 expression_score=0.0,
#                 background_score=0.0,
#                 overall_score=0.0,
#                 face_size_pass=False,
#                 expression_pass=False,
#                 background_pass=False,
#                 overall_pass=False,
#                 # New scores with default values
#                 face_position_score=0.0,
#                 face_pose_score=0.0,
#                 face_occlusion_score=0.0,
#                 lips_open_score=0.0,
#                 eyes_open_score=0.0,
#                 eye_color_score=0.0,
#                 eye_direction_score=0.0,
#                 shadow_score=0.0,
#                 exposure_score=0.0,
#                 sharpness_score=0.0,
#                 shoulder_score=0.0,
#                 # New pass/fail indicators with default values
#                 face_position_pass=False,
#                 face_pose_pass=False,
#                 face_occlusion_pass=False,
#                 lips_open_pass=False,
#                 eyes_open_pass=False,
#                 eye_color_pass=False,
#                 eye_direction_pass=False,
#                 shadow_pass=False,
#                 exposure_pass=False,
#                 sharpness_pass=False,
#                 shoulder_pass=False,
#                 resultcode="NO_FACE",
#                 resultmessage="No face detected in image"
#             )
        
#         landmarks = mesh_results.multi_face_landmarks[0].landmark
        
#         # 1. Face Position Check
#         face_center_x = np.mean([landmarks[i].x for i in [1, 152, 454]])
#         face_center_y = np.mean([landmarks[i].y for i in [1, 152, 454]])
#         face_position_score = 1.0 - (abs(0.5 - face_center_x) + abs(0.5 - face_center_y))
        
#         # 2. Face Pose Check
#         nose_tip = landmarks[4]
#         left_eye = landmarks[33]
#         right_eye = landmarks[263]
#         pose_score = 1.0 - abs(left_eye.z - right_eye.z)
        
#         # 3. Expression Check (using mouth landmarks)
#         mouth_height = abs(landmarks[13].y - landmarks[14].y)
#         mouth_width = abs(landmarks[78].x - landmarks[308].x)
#         expression_score = (1.0 - mouth_height * 2) * (1.0 - abs(mouth_width - 0.5) * 2)
        
#         # 4. Face Occlusion Check
#         # Using face mesh coverage
#         occlusion_score = len([l for l in landmarks if l.visibility > 0.9]) / len(landmarks)
        
#         # 5. Lips Close Check
#         lips_close_score = 1.0 - abs(landmarks[13].y - landmarks[14].y)
        
#         # 6. Skin Tone Check
#         face_roi = image[int(face_center_y*h-h/4):int(face_center_y*h+h/4),
#                         int(face_center_x*w-w/4):int(face_center_x*w+w/4)]
#         skin_tone_score = cv2.mean(face_roi)[0] / 255.0
        
#         # 7 & 8. Eye Open and Red Eye Check
#         left_eye_height = abs(landmarks[159].y - landmarks[145].y)
#         right_eye_height = abs(landmarks[386].y - landmarks[374].y)
#         eye_open_score = (left_eye_height + right_eye_height) / 2
        
#         # Red eye check using HSV color space
#         hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#         red_mask = cv2.inRange(hsv, (0,50,50), (10,255,255))
#         red_eye_score = 1.0 - (np.sum(red_mask) / (h * w * 255))
        
#         # 9. Eye Direction Check
#         left_pupil = landmarks[468]
#         right_pupil = landmarks[473]
#         eye_direction_score = 1.0 - (abs(left_pupil.x - left_eye.x) + abs(right_pupil.x - right_eye.x))
        
#         # 10 & 11. Shadow and Background Check
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         background_std = np.std(gray)
#         background_score = 1.0 - (background_std / 128.0)
#         shadow_score = 1.0 - (np.std(gray) / 128.0)
        
#         # 12. Exposure Check
#         exposure_score = 1.0 - abs(np.mean(gray) - 128) / 128
        
#         # 13. Sharpness Check
#         laplacian = cv2.Laplacian(gray, cv2.CV_64F)
#         sharpness_score = min(1.0, np.std(laplacian) / 100)
        
#         # 14. Shoulder Check
#         shoulder_left = landmarks[93]
#         shoulder_right = landmarks[323]
#         shoulder_score = 1.0 - abs(shoulder_left.y - shoulder_right.y)
        
#         # 15. Face Size Proportion Check
#         face_height = abs(landmarks[10].y - landmarks[152].y)
#         face_width = abs(landmarks[234].x - landmarks[454].x)
#         height_proportion = face_height * 100
#         width_proportion = face_width * 100
#         size_score = 1.0 if (71 <= height_proportion <= 80 and 60 <= width_proportion <= 75) else 0.0
        
#         # Calculate overall score
#         scores = [face_position_score, pose_score, expression_score, occlusion_score,
#                  lips_close_score, skin_tone_score, eye_open_score, red_eye_score,
#                  eye_direction_score, shadow_score, background_score, exposure_score,
#                  sharpness_score, shoulder_score]
#         overall_score = sum(scores) / len(scores)
        
#         # Determine pass/fail for each metric with thresholds
#         face_position_pass = face_position_score > 0.8  # Face should be centered
#         face_pose_pass = pose_score > 0.8  # Face should be front-facing
#         face_occlusion_pass = occlusion_score > 0.9  # No occlusion
#         lips_open_pass = lips_close_score > 0.8  # Mouth should be closed
#         eyes_open_pass = eye_open_score > 0.7  # Eyes should be open
#         eye_color_pass = red_eye_score > 0.9  # No red eye
#         eye_direction_pass = eye_direction_score > 0.8  # Eyes looking forward
#         shadow_pass = shadow_score > 0.8  # No significant shadows
#         exposure_pass = exposure_score > 0.7  # Proper exposure
#         sharpness_pass = sharpness_score > 0.7  # Image should be sharp
#         shoulder_pass = shoulder_score > 0.8  # Shoulders should be level
        
#         return ICAOCheckOutput(
#             base64image=input_data.base64image,
#             face_size_score=round(size_score, 2),
#             expression_score=round(expression_score, 2),
#             background_score=round(background_score, 2),
#             overall_score=round(overall_score, 2),
#             face_size_pass=size_score > 0.8,
#             expression_pass=expression_score > 0.8,
#             background_pass=background_score > 0.8,
#             overall_pass=overall_score > 0.8,
#             # New scores
#             face_position_score=round(face_position_score, 2),
#             face_pose_score=round(pose_score, 2),
#             face_occlusion_score=round(occlusion_score, 2),
#             lips_open_score=round(lips_close_score, 2),
#             eyes_open_score=round(eye_open_score, 2),
#             eye_color_score=round(red_eye_score, 2),
#             eye_direction_score=round(eye_direction_score, 2),
#             shadow_score=round(shadow_score, 2),
#             exposure_score=round(exposure_score, 2),
#             sharpness_score=round(sharpness_score, 2),
#             shoulder_score=round(shoulder_score, 2),
#             # New pass/fail indicators
#             face_position_pass=face_position_pass,
#             face_pose_pass=face_pose_pass,
#             face_occlusion_pass=face_occlusion_pass,
#             lips_open_pass=lips_open_pass,
#             eyes_open_pass=eyes_open_pass,
#             eye_color_pass=eye_color_pass,
#             eye_direction_pass=eye_direction_pass,
#             shadow_pass=shadow_pass,
#             exposure_pass=exposure_pass,
#             sharpness_pass=sharpness_pass,
#             shoulder_pass=shoulder_pass,
#             resultcode="SUCCESS",
#             resultmessage="Passport photo check completed successfully"
#         )
#     except Exception as e:
#         logger.error(f"Error checking passport photo: {str(e)}")
#         return ICAOCheckOutput(
#             base64image=input_data.base64image,
#             face_size_score=0.0,
#             expression_score=0.0,
#             background_score=0.0,
#             overall_score=0.0,
#             face_size_pass=False,
#             expression_pass=False,
#             background_pass=False,
#             overall_pass=False,
#             # New scores with default values
#             face_position_score=0.0,
#             face_pose_score=0.0,
#             face_occlusion_score=0.0,
#             lips_open_score=0.0,
#             eyes_open_score=0.0,
#             eye_color_score=0.0,
#             eye_direction_score=0.0,
#             shadow_score=0.0,
#             exposure_score=0.0,
#             sharpness_score=0.0,
#             shoulder_score=0.0,
#             # New pass/fail indicators with default values
#             face_position_pass=False,
#             face_pose_pass=False,
#             face_occlusion_pass=False,
#             lips_open_pass=False,
#             eyes_open_pass=False,
#             eye_color_pass=False,
#             eye_direction_pass=False,
#             shadow_pass=False,
#             exposure_pass=False,
#             sharpness_pass=False,
#             shoulder_pass=False,
#             resultcode="ERROR",
#             resultmessage=str(e)
#         )


