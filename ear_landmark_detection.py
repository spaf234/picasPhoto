import cv2
import numpy as np
import os
import mediapipe as mp

import traceback
from datetime import datetime



# https://nalinc.github.io/blog/2018/skin-detection-python-opencv/
# https://medium.com/swlh/human-skin-color-classification-using-the-threshold-classifier-rgb-ycbcr-hsv-python-code-d34d51febdf8
# https://medium.com/@neerajpoladen/find-skin-using-hsvvalues-with-opencv-59d066186c7b

def load_ear_model():
    """
    Initialize MediaPipe Face Mesh for ear detection
    Returns:
        MediaPipe Face Mesh object
    """
    try:
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5,
            refine_landmarks=True,
            min_tracking_confidence=0.5
        )
        return face_mesh
    except Exception as e:
        print(f"Error initializing ear detection: {str(e)}")
        return None
    
    
    
def check_ear_visibility(landmarks, ear_landmarks, image_width, image_height, image, image_skinColor_in_earRegion_mask, ear_type, threshold_skinColor_earRegion=0.0002):
    """
    Check if ear is visible based on landmark positions and skin color analysis
    Args:
        landmarks: MediaPipe face landmarks
        ear_landmarks: List of landmark indices for ear detection
        image_width: Width of the image
        image_height: Height of the image
        image: Original image in BGR format
        ear_type: 'left' or 'right' to indicate which ear
        threshold: Minimum area threshold
    Returns:
        Tuple of (is_visible, confidence)
    """
    
    try:
        # Get original ear landmark points and draw green circles with indices
        original_points = []
        for idx in ear_landmarks:
            x = landmarks[idx].x * image_width
            y = landmarks[idx].y * image_height
            original_points.append((x, y))
            # Draw green dot and landmark index
            cv2.circle(image, (int(x), int(y)), 2, (0, 255, 0), -1)
            cv2.putText(image, str(idx), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)

        # Convert original points to numpy array for polygon operations
        original_points = np.array(original_points, dtype=np.float32)
        
        # Create analysis polygon points with shifted copies
        analysis_points = []
        if ear_type == 'left':
            # Create shifted points (-12 pixels in x direction)
            shifted_points = original_points.copy()
            shifted_points[:, 0] -= 12
            
            # Combine original and shifted points to create analysis region
            analysis_points = np.vstack((original_points, shifted_points[::-1]))  # Reverse shifted points for proper polygon
            color = (0, 0, 255)  # Red for left ear
        else:  # right ear
            # Create shifted points (+12 pixels in x direction) 
            shifted_points = original_points.copy()
            shifted_points[:, 0] += 12
            
            # Combine original and shifted points to create analysis region
            analysis_points = np.vstack((original_points, shifted_points[::-1]))  # Reverse shifted points for proper polygon
            color = (255, 0, 0)  # Blue for right ear

        # Convert points to integer for drawing
        analysis_points_int = analysis_points.astype(np.int32)
        
        # Create mask for analysis region
        mask = np.zeros((int(image_height), int(image_width)), dtype=np.uint8)
        cv2.fillPoly(mask, [analysis_points_int], 255)
        
        # Draw analysis region outline
        cv2.polylines(image, [analysis_points_int], True, color, 2)
        
        # Get bounding rectangle of the polygon
        x, y, w, h = cv2.boundingRect(analysis_points_int)
        
        # Extract region using the bounding rectangle and mask
        roi = image[y:y+h, x:x+w]
        mask_roi = mask[y:y+h, x:x+w]
        analyze_region = cv2.bitwise_and(roi, roi, mask=mask_roi)
        
        if np.sum(mask) == 0:
            return False, 0.0
            
        # Convert to YCrCb color space for better skin detection
        ycrcb_region = cv2.cvtColor(analyze_region, cv2.COLOR_BGR2YCrCb)
        
        # https://nalinc.github.io/blog/2018/skin-detection-python-opencv/
        # min_YCrCb = np.array([0, 133, 77], np.uint8)
        # max_YCrCb = np.array([235, 173, 127], np.uint8)
        
        
        # https://www.mikekohn.net/file_formats/yuv_rgb_converter.php  or  use Gimp
        # Define skin color ranges in YCrCb
        # These values are typical for Asian skin tones
        #min_YCrCb = np.array([0, 135, 85], np.uint8)
        #max_YCrCb = np.array([255, 180, 135], np.uint8)
        
        # ordinary skin color
        #min_YCrCb = np.array([80, 133, 77], np.uint8)
        #max_YCrCb = np.array([255, 173, 127], np.uint8)
        
        # Dark Brown 을 포함할경우 ear가 아니라 Brown hair를 포함하는 경우가 생기므로 skin color 범위를 좁혀준다.
        # Blonde hair 의 경우 ear 영역에서 비율이 높아질수 있는 단점.
        #min_YCrCb = np.array([170, 145, 115], np.uint8)
        #max_YCrCb = np.array([214, 180, 135], np.uint8)
        min_YCrCb = np.array([70, 140, 105], np.uint8)
        max_YCrCb = np.array([255, 185, 135], np.uint8)
        
        # Create skin mask
        skin_mask = cv2.inRange(ycrcb_region, min_YCrCb, max_YCrCb)
        
        
        #---------------------------------------------------------------------------------------
        # HSV + YCrCb detect merge ( bitwiseAnd)  ------  start ------
        # https://github.com/CHEREF-Mehdi/SkinDetection
        # 추가기능 테스트트
        #----------------------------------------------------------------------------------------
        
        
        #converting from gbr to hsv color space
        img_HSV = cv2.cvtColor(analyze_region, cv2.COLOR_BGR2HSV)
        #skin color range for hsv color space 
        #HSV_mask = cv2.inRange(img_HSV, (0, 15, 0), (17,170,255)) 
        HSV_mask = cv2.inRange(img_HSV, (0, 15, 50), (26,170,255)) 
        HSV_mask = cv2.morphologyEx(HSV_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

        #converting from gbr to YCbCr color space
        img_YCrCb = cv2.cvtColor(analyze_region, cv2.COLOR_BGR2YCrCb)
        #skin color range for hsv color space 
        YCrCb_mask = cv2.inRange(img_YCrCb, (70, 140, 85), (255,185,135)) 
        #YCrCb_mask = cv2.morphologyEx(YCrCb_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
        YCrCb_mask = cv2.morphologyEx(YCrCb_mask, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))

        #merge skin detection (YCbCr and hsv)
        #global_mask=cv2.bitwise_and(YCrCb_mask,HSV_mask)
        global_mask=cv2.bitwise_or(YCrCb_mask,HSV_mask)
        global_mask=cv2.medianBlur(global_mask,3)
        #global_mask = cv2.morphologyEx(global_mask, cv2.MORPH_OPEN, np.ones((4,4), np.uint8))
        global_mask = cv2.morphologyEx(global_mask, cv2.MORPH_CLOSE, np.ones((4,4), np.uint8))
        
  
        
        # Calculate percentage of skin pixels in the masked region
        total_pixels_merge = np.count_nonzero(mask)
        skin_pixels_merge = np.count_nonzero(cv2.bitwise_and(global_mask, mask_roi))
        skin_ratio_merge = skin_pixels_merge / total_pixels_merge if total_pixels_merge > 0 else 0
        
        
              
        # Create skin color visualization mask
        skin_mask_full_global = np.zeros((image_height, image_width), dtype=np.uint8)
        skin_mask_full_global[y:y+h, x:x+w] = global_mask
        
        
        # print(datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f'),  datetime.now().timestamp() * 1000
        #cv2.imshow("temp/skin_mask_full_global"+ str( datetime.now().strftime('%Y-%m-%d_%H_%M_%S_%f')) +".jpg",skin_mask_full_global)   
        cv2.imwrite("temp/skin_mask_full_global"+ str( datetime.now().strftime('%Y-%m-%d_%H_%M_%S_%f')) +".jpg",skin_mask_full_global)        
        
        
        #---------------------------------------------------------------------------------------
        # HSV + YCrCb detect merge ( bitwiseAnd)  ------  end ------
        # bitwise_and로 merge하면 더 적게 detect된다. bitwise_or로 하면 더 많이 detect된다 . 필요시 color 조정필요.
        #---------------------------------------------------------------------------------------
        
        # Calculate percentage of skin pixels in the masked region
        total_pixels = np.count_nonzero(mask)
        skin_pixels = np.count_nonzero(cv2.bitwise_and(skin_mask, mask_roi))
        skin_ratio = skin_pixels / total_pixels if total_pixels > 0 else 0
        
        
        print(f"skin ratio :{skin_ratio}, skin_ratio_merge: {skin_ratio_merge}" )
        
        # Calculate confidence based on skin color match
        #confidence_skinPercent_in_earRegion = skin_ratio
        confidence_skinPercent_in_earRegion = skin_ratio_merge
        
        # Check minimum area requirement
        min_area = (image_width * image_height) * threshold_skinColor_earRegion
        area = cv2.contourArea(original_points.reshape(-1, 1, 2))
        
        # area : 왼쪽, 오른쪽 landmark에서 12pixel 오른쪽, 왼쪽으로 이동한 점들을 연결하여 만든 다각형의 면적
        # min_area: 전체 이미지 width * height * threshold_skinColor_earRegion 값을 곱한 값 ( 0.02 : 전체 이미지 영역중 2% 이상 차지하는 경우 귀가 보이는 것으로 판단 )
        # earRegion 영역에서 skin color 비율이 24% 이상인 경우 귀가 보이는 것으로 판단
        is_visible = area > min_area and confidence_skinPercent_in_earRegion > 0.24

        # Create skin color visualization mask
        skin_mask_full = np.zeros((image_height, image_width), dtype=np.uint8)
        skin_mask_full[y:y+h, x:x+w] = skin_mask
        
        # Draw magenta contours around skin regions
        contours, _ = cv2.findContours(skin_mask_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image_skinColor_in_earRegion_mask, contours, -1, (255, 0, 255), 2)  # Magenta color (255,0,255)
        
        #cv2.imwrite('image_skinColor_in_earRegion_mask.png', image_skinColor_in_earRegion_mask)
    
    except Exception as e:
            # or
            # use sys ( import sys)
            #exc_type, exc_obj, exc_tb = sys.exc_info()
            #fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            #print(f'file name: {str(fname)}')
            #print(f'error type: {str(exc_type)}')
            #print(f'error msg: {str(e)}')
            #print(f'line number: {str(exc_tb.tb_lineno)}')
            # or 
            # use traceback( import traceback)
            print(traceback.format_exc())
    
    return is_visible, confidence_skinPercent_in_earRegion

def detect_ears(image, face_mesh, image_skinColor_in_earRegion_mask):
    """
    Detect ears using face landmarks and contour analysis
    Args:
        image: Input image (BGR format)
        face_mesh: MediaPipe Face Mesh object
    Returns:
        Dictionary containing ear detection results and counts
    """
    if face_mesh is None:
        return {
            'results': [],
            'total_count': 0,
            'left_count': 0,
            'right_count': 0,
            'left_confidence': 0.0,
            'right_confidence': 0.0
        }
        
    results = []
    left_count = 0
    right_count = 0
    left_confidence = 0.0
    right_confidence = 0.0
    
    try:
        # Convert to RGB for MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        
        # Process image with face mesh
        mesh_results = face_mesh.process(cv2.resize(image_rgb, (w, h)))
        
        if not mesh_results.multi_face_landmarks:
            return {
                'results': [],
                'total_count': 0,
                'left_count': 0,
                'right_count': 0,
                'left_confidence': 0.0,
                'right_confidence': 0.0
            }
            
        # Get face landmarks
        landmarks = mesh_results.multi_face_landmarks[0].landmark
        
        # Face의 가로, 세로 비율확인( 얼굴이 위아래로 긴 형태인지, 좌우로 긴 형태인지 확인 )
        # Get min/max coordinates to create bounding box
        x_coords = [landmark.x for landmark in landmarks]
        y_coords = [landmark.y for landmark in landmarks]
        
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        # Convert normalized coordinates to pixel coordinates
        x_min_px = int(x_min * w)
        x_max_px = int(x_max * w)
        y_min_px = int(y_min * h) 
        y_max_px = int(y_max * h)
        
        # Draw bounding box
        cv2.rectangle(image, (x_min_px, y_min_px), (x_max_px, y_max_px), (0, 255, 0), 2)
        
        # Calculate aspect ratio
        bbox_width = x_max - x_min
        bbox_height = y_max - y_min
        
        
        aspect_ratio = bbox_height / bbox_width
        
        print(f"aspect_ratio: {aspect_ratio} =  {bbox_height} / {bbox_width}")
        
        
        # Define ear region landmarks
        #left_ear_landmarks = [162, 127,234, 93, 132, 58, 454]  # Left ear landmarks
        #right_ear_landmarks = [389,356, 454, 323, 361, 288, 234]  # Right ear landmarks
        
        #  얼굴이 좌우로 긴 형태인 경우나 보통비율인 경우
        if aspect_ratio < 1.12:
            # 왼쪽 귀 영역 확인
            left_ear_landmarks = [162, 127,234, 93, 132]  # Left ear landmarks
            right_ear_landmarks = [389,356, 454, 323, 361]  # Right ear landmarks
        else:
            # 얼굴이 위아래로 긴 형태인 경우
            left_ear_landmarks = [162, 127,234, 93, 132, 58 ]  # Left ear landmarks
            right_ear_landmarks = [389,356, 454, 323, 361, 288]  # Right ear landmarks

        # Check visibility for each ear
        left_visible, left_conf = check_ear_visibility(landmarks, left_ear_landmarks, w, h, image, image_skinColor_in_earRegion_mask, 'left')
        right_visible, right_conf = check_ear_visibility(landmarks, right_ear_landmarks, w, h, image, image_skinColor_in_earRegion_mask, 'right')

        if left_visible:
            # Process left ear
            left_ear_points = []
            for idx in left_ear_landmarks:
                x = int(landmarks[idx].x * w)
                y = int(landmarks[idx].y * h)
                left_ear_points.append((x, y))
                
            results.append({
                'landmarks': np.array(left_ear_points),
                'confidence': left_conf,
                'type': 'left'
            })
            left_count = 1
            left_confidence = left_conf
        
        if right_visible:
            # Process right ear
            right_ear_points = []
            for idx in right_ear_landmarks:
                x = int(landmarks[idx].x * w)
                y = int(landmarks[idx].y * h)
                right_ear_points.append((x, y))
                
            results.append({
                'landmarks': np.array(right_ear_points),
                'confidence': right_conf,
                'type': 'right'
            })
            right_count = 1
            right_confidence = right_conf
            
    except Exception as e:
        print(f"Error during ear detection: {str(e)}")
    
    return {
        'results': results,
        'total_count': len(results),
        'left_count': left_count,
        'right_count': right_count,
        'left_confidence': left_confidence,
        'right_confidence': right_confidence
    } 