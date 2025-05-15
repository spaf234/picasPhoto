import gradio as gr
import base64
import requests
import io
from PIL import Image
import json

# Base URL for the FastAPI service
BASE_URL = "http://localhost:8801"

def encode_image_to_base64(image):
    if isinstance(image, str):  # If image is already a file path
        return image
    
    # Convert to base64
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

def decode_base64_to_image(base64_string):
    if not base64_string:
        return None
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))
    return image

def detect_landmarks(image):
    if image is None:
        return None, None, None, None, None, 0, 0, 0.0, 0, 0.0, "ERROR", "No image provided"
    
    # Encode image to base64
    base64_image = encode_image_to_base64(image)
    
    # Prepare request data
    data = {
        "base64image": base64_image,
        "output_imgType": "png"
    }
    
    # Make request to FastAPI endpoint
    response = requests.post(f"{BASE_URL}/detect_landmarks", json=data)
    result = response.json()
    
    # Decode base64 images
    face_image = decode_base64_to_image(result["base64image_face"])
    face_dot_image = decode_base64_to_image(result["base64image_faceDot"])
    face_dlib_image = decode_base64_to_image(result["base64image_faceDLib"])
    face_dlib_numbered_image = decode_base64_to_image(result["base64image_faceDLib_68_number"])
    ear_detect_image = decode_base64_to_image(result["base64image_ear_detect"])
    ear_skinColor_detect_image = decode_base64_to_image(result["base64image_ear_skinColor_detect"])
    
    return (face_image, 
            face_dot_image, 
            face_dlib_image, 
            face_dlib_numbered_image,
            ear_detect_image,
            ear_skinColor_detect_image,
            result["ear_detected_count"],
            result["ear_left_count"],
            result["ear_left_confidence"],
            result["ear_right_count"],
            result["ear_right_confidence"],
            result["resultcode"], 
            result["resultmessage"])

def align_face(image):
    if image is None:
        return None, [], "ERROR", "No image provided"
    
    # Encode image to base64
    base64_image = encode_image_to_base64(image)
    
    # Prepare request data
    data = {
        "base64image": base64_image,
        "output_imgType": "png"
    }
    
    # Make request to FastAPI endpoint
    response = requests.post(f"{BASE_URL}/align_face", json=data)
    result = response.json()
    
    # Decode base64 image
    aligned_image = decode_base64_to_image(result["base64image_align"])
    bg_mask_image = decode_base64_to_image(result["base64image_bg_mask"])
    
    return (aligned_image, 
            bg_mask_image,
            result["background_color"], 
            result["resultcode"], 
            result["resultmessage"])

def compare_faces(image1, image2):
    if image1 is None or image2 is None:
        return 0.0, "ERROR", "Both images must be provided"
    
    # Encode images to base64
    base64_image1 = encode_image_to_base64(image1)
    base64_image2 = encode_image_to_base64(image2)
    
    # Prepare request data
    data = {
        "base64image1": base64_image1,
        "base64image2": base64_image2,
        "filename1": "image1.png",
        "filename2": "image2.png"
    }
    
    # Make request to FastAPI endpoint
    response = requests.post(f"{BASE_URL}/face-similarity", json=data)
    result = response.json()
    
    return (result["similarity_score"], 
            result["resultcode"], 
            result["resultmessage"])

def check_pspt_photo_quality(image, output_type="jpg"):
    if image is None:
        return None, None, None, None, None, None, 0, 0, 0.0, 0, 0.0, None, 0.0, "ERROR", "No image provided"
    
    try:
        # Encode image to base64
        base64_image = encode_image_to_base64(image)
        
        # Prepare request data
        data = {
            "base64image": base64_image,
            "output_imgType": output_type
        }
        
        # Make request to FastAPI endpoint
        response = requests.post(f"{BASE_URL}/check_pspt_photo_quality", json=data)
        result = response.json()
        
        # Decode base64 images
        face_detect_image = decode_base64_to_image(result["base64image_face_detect"])
        head_detect_image = decode_base64_to_image(result["base64image_head_detect"])
        bg_removed_image = decode_base64_to_image(result["base64image_bg_removed"])
        bg_removed_canny_image = decode_base64_to_image(result["base64image_bg_removed_canny"])
        bg_removed_canny_upd_crown = decode_base64_to_image(result["base64image_bg_removed_canny_upd_crown"])
        ear_detect_image = decode_base64_to_image(result["base64image_ear_detect"])
        ear_skinColor_detect_image = decode_base64_to_image(result["base64image_ear_skinColor_detect"])
        
        # Create scores data
        scores_data = [
            ["Face Height Proportion", result["face_height_proportion"], ""],
            ["Face Width Proportion", result["face_width_proportion"], ""],
            ["Head Height Proportion", result["head_height_proportion"], ""],
            ["Head Width Proportion", result["head_width_proportion"], ""],
            ["Blur Score", result["blur_score"], ""],
            ["Pixelation Score", result["pixelation_score"], ""],
            ["White Noise Score", result["white_noise_score"], ""],
            ["Contrast Score", result["contrast_score"], ""],
            ["General Illumination Score", result["general_illumination_score"], ""],
            ["Face Position Score", result["face_position_score"], ""],
            ["Face Pose Score", result["face_pose_score"], ""],
            ["Expression Score", result["expression_score"], ""],
            ["Eyes Open Score", result["eyes_open_score"], ""],
            ["Eyes Direction Score", result["eyes_direction_score"], ""],
            ["Mouth Open Score", result["mouth_open_score"], ""],
            ["Hair Over Face Score", result["hair_over_face_score"], ""],
            ["Sunglasses Score", result["sunglasses_score"], ""],
            ["Glasses Reflection Score", result["glasses_reflection_score"], ""],
            ["Glasses Frame Score", result["glasses_frame_score"], ""],
            ["Glasses Covering Score", result["glasses_covering_score"], ""],
            ["Hat Score", result["hat_score"], ""],
            ["Veil Score", result["veil_score"], ""],
            ["Skin Color Score", result["skin_color_score"], ""],
            ["Red Eyes Score", result["red_eyes_score"], ""],
            ["Skin Reflection Score", result["skin_reflection_score"], ""],
            ["Shadow Face Score", result["shadow_face_score"], ""],
            ["Shadow Background Score", result["shadow_background_score"], ""],
            ["Background Uniformity Score", result["background_uniformity_score"], ""],
            ["Ink Marks Score", result["ink_marks_score"], ""],
            ["Other Faces Score", result["other_faces_score"], ""]
        ]
        
        overall_score = result["overall_score"]
        
        return (face_detect_image,
                head_detect_image,
                bg_removed_image,
                bg_removed_canny_image,
                bg_removed_canny_upd_crown,
                ear_detect_image,
                ear_skinColor_detect_image,
                result["ear_detected_count"],
                result["ear_left_count"],
                result["ear_left_confidence"],
                result["ear_right_count"],
                result["ear_right_confidence"],
                scores_data,
                overall_score,
                result["resultcode"],
                result["resultmessage"])
                
    except Exception as e:
        return (None,
                None,
                None,
                None,
                None,
                None,
                None,
                0,
                0,
                0.0,
                0,
                0.0,
                None,
                0.0,
                "ERROR",
                f"Error processing passport photo: {str(e)}")

# Create Gradio interface
with gr.Blocks(title="Face Detection Services") as demo:
    gr.Markdown("# Face Detection Services")
    
    with gr.Tab("Face Landmarks Detection"):
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(type="pil", label="Input Image")
            with gr.Column():
                output_image = gr.Image(type="pil", label="Face Mesh")
                output_dot = gr.Image(type="pil", label="Face Landmarks (Dots)")
                output_dlib = gr.Image(type="pil", label="Face Landmarks (DLib)")
                output_dlib_numbered = gr.Image(type="pil", label="Face Landmarks (DLib Number)")
                output_ear = gr.Image(type="pil", label="Ear Detection")
                output_ear_skinColor = gr.Image(type="pil", label="Ear Skin Color Detection")
                ear_count = gr.Number(label="Total Ear Count")
                ear_left_count = gr.Number(label="Left Ear Count")
                ear_left_conf = gr.Number(label="Left Ear Confidence")
                ear_right_count = gr.Number(label="Right Ear Count")
                ear_right_conf = gr.Number(label="Right Ear Confidence")
        result_code = gr.Textbox(label="Result Code")
        result_message = gr.Textbox(label="Result Message")
        detect_btn = gr.Button("Detect Landmarks")
        detect_btn.click(
            fn=detect_landmarks,
            inputs=[input_image],
            outputs=[output_image, output_dot, output_dlib, output_dlib_numbered, 
                    output_ear, output_ear_skinColor, ear_count, ear_left_count, ear_left_conf, 
                    ear_right_count, ear_right_conf, result_code, result_message]
        )
    with gr.Tab("Face Alignment"):
        with gr.Row():
            with gr.Column():
                align_input = gr.Image(type="pil", label="Input Image")
            with gr.Column():
                align_output = gr.Image(type="pil", label="Aligned Face")
                bg_mask_output = gr.Image(type="pil", label="Background Mask")
                bg_color = gr.JSON(label="Background Color")
        align_code = gr.Textbox(label="Result Code")
        align_message = gr.Textbox(label="Result Message")
        align_btn = gr.Button("Align Face")
        align_btn.click(
            fn=align_face,
            inputs=[align_input],
            outputs=[align_output, bg_mask_output, bg_color, align_code, align_message]
        )
    with gr.Tab("Face Similarity"):
        with gr.Row():
            with gr.Column():
                face1_input = gr.Image(type="pil", label="Face 1")
                face2_input = gr.Image(type="pil", label="Face 2")
            with gr.Column():
                similarity_score = gr.Number(label="Similarity Score")
        compare_code = gr.Textbox(label="Result Code")
        compare_message = gr.Textbox(label="Result Message")
        compare_btn = gr.Button("Compare Faces")
        compare_btn.click(
            fn=compare_faces,
            inputs=[face1_input, face2_input],
            outputs=[similarity_score, compare_code, compare_message]
        )
    with gr.Tab("Passport Photo Quality Check"):
        with gr.Row():
            with gr.Column():
                pspt_input = gr.Image(type="pil", label="Passport Photo")
                output_type = gr.Dropdown(
                    choices=["jpg", "png", "bmp"], 
                    value="jpg",
                    label="Output Image Type"
                )
            with gr.Column():
                bg_removed_output = gr.Image(type="pil", label="Background Removed")
                face_detect_output = gr.Image(type="pil", label="Face Detection")  
            with gr.Column():
                bg_removed_canny_output = gr.Image(type="pil", label="Background Removed with Edges")           
                head_detect_output = gr.Image(type="pil", label="Head Detection")
                ear_detect_output = gr.Image(type="pil", label="Ear Detection")
                ear_skinColor_detect_output = gr.Image(type="pil", label="Ear Skin Color Detection")
                ear_count = gr.Number(label="Total Ear Count")
                ear_left_count = gr.Number(label="Left Ear Count")
                ear_left_conf = gr.Number(label="Left Ear Confidence")
                ear_right_count = gr.Number(label="Right Ear Count")
                ear_right_conf = gr.Number(label="Right Ear Confidence")
        
        with gr.Row():
            with gr.Column():
                scores_df = gr.Dataframe(
                    headers=["Metric", "Score", "Pass/Fail"],
                    label="Quality Scores"
                )
            with gr.Column():
                overall_score = gr.Number(label="Overall Score")
                bg_removed_canny_upd_crown_output = gr.Image(type="pil", label="Updated Crown Point")
                
        pspt_code = gr.Textbox(label="Result Code")
        pspt_message = gr.Textbox(label="Result Message")
        check_btn = gr.Button("Check Photo Quality")
        check_btn.click(
            fn=check_pspt_photo_quality,
            inputs=[pspt_input, output_type],
            outputs=[face_detect_output, head_detect_output, bg_removed_output, 
                    bg_removed_canny_output, bg_removed_canny_upd_crown_output,
                    ear_detect_output, ear_skinColor_detect_output, ear_count, ear_left_count, ear_left_conf,
                    ear_right_count, ear_right_conf, scores_df, overall_score,
                    pspt_code, pspt_message]
        )

if __name__ == "__main__":
    demo.launch( server_port=8800) 
    