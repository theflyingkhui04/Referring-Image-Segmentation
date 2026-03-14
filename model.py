import numpy as np
import cv2
import torch
from transformers import AutoProcessor, CLIPSegForImageSegmentation
import gradio as gr
import os

# ========================================
# CẤU HÌNH & LOAD MODEL
# ========================================
student_dir = "clipseg_student/hf_best"
MODEL_ID = "CIDAS/clipseg-rd64-refined"
weight_path = './ckpt_best.pt'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"🖥️  Device: {DEVICE}")
print("📥 Loading model...")

processor = AutoProcessor.from_pretrained(
    student_dir if os.path.exists(student_dir) else MODEL_ID
)

model = CLIPSegForImageSegmentation.from_pretrained(
    student_dir if os.path.exists(student_dir) else MODEL_ID
).to(DEVICE)

if os.path.exists(weight_path):
    print(f"📦 Loading checkpoint: {weight_path}")
    checkpoint = torch.load(weight_path, map_location='cpu', weights_only=True)
    state_dict = checkpoint.get('model', checkpoint)
    model.load_state_dict(state_dict, strict=False)
    print("✅ Model loaded!")

model.eval()

# ========================================
# HÀM XỬ LÝ
# ========================================
def preprocess_text(text):
    if text is None or text.strip() == "":
        return ""
    return ' '.join(text.strip().split())

def segment_image(image, text, threshold):
    """Hàm chính để segment ảnh"""
    if image is None:
        return None, None, "❌ Chưa upload ảnh!"
    
    if not text or text.strip() == "":
        return image, None, "⚠️ Chưa nhập text mô tả!"
    
    # Chuyển RGB -> BGR (Gradio trả về RGB)
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    orig_h, orig_w = image_bgr.shape[:2]
    
    # Xử lý text
    cleaned_text = preprocess_text(text)
    
    # Process inputs
    text_inputs = processor.tokenizer(
        [cleaned_text],
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    
    image_inputs = processor.image_processor(
        images=[image],  # Gradio image đã là RGB
        return_tensors="pt"
    )
    
    inputs = {**text_inputs, **image_inputs}
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    
    # Inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        
        if logits.ndim == 4:
            logits = logits[0, 0]
        elif logits.ndim == 3:
            logits = logits[0]
        
        probs = torch.sigmoid(logits).cpu().numpy()
    
    # Postprocess
    mask_resized = cv2.resize(probs, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    mask_binary = (mask_resized >= threshold).astype(np.uint8) * 255
    
    # Tạo overlay
    overlay = image.copy()
    mask_bool = mask_binary > 0
    
    if mask_bool.any():
        overlay[mask_bool] = (
            np.array([255, 0, 0]) * 0.5 + overlay[mask_bool] * 0.5
        ).astype(np.uint8)
        
        # Vẽ contour
        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        cv2.drawContours(overlay_bgr, contours, -1, (0, 255, 0), 2)
        overlay = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
    
    # Thống kê
    coverage = (mask_binary > 0).sum() / mask_binary.size
    info = f"""
✅ **Segmentation completed!**
- Text: "{text}"
- Threshold: {threshold}
- Coverage: {coverage*100:.2f}%
- Masked pixels: {(mask_binary > 0).sum():,}/{mask_binary.size:,}
    """
    
    return overlay, mask_binary, info

# ========================================
# GRADIO INTERFACE
# ========================================
with gr.Blocks(title="CLIPSeg Inference") as demo:
    gr.Markdown("# 🎯 CLIPSeg Image Segmentation")
    gr.Markdown("Upload ảnh và mô tả đối tượng cần segment")
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="📷 Input Image", type="numpy")
            input_text = gr.Textbox(
                label="📝 Text Description",
                placeholder="VD: girl, boy on the right, tree...",
                value="girl"
            )
            threshold_slider = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.5,
                step=0.05,
                label="🎚️ Threshold"
            )
            submit_btn = gr.Button("🚀 Segment", variant="primary")
        
        with gr.Column():
            output_overlay = gr.Image(label="🖼️ Overlay Result")
            output_mask = gr.Image(label="🎭 Binary Mask")
            output_info = gr.Markdown()
    
    # Ví dụ
    gr.Examples(
        examples=[
            ["./3.jpg", "girl", 0.5],
            ["./3.jpg", "boy on the right", 0.5],
            ["./3.jpg", "tree", 0.4],
        ],
        inputs=[input_image, input_text, threshold_slider],
    )
    
    submit_btn.click(
        fn=segment_image,
        inputs=[input_image, input_text, threshold_slider],
        outputs=[output_overlay, output_mask, output_info]
    )

if __name__ == "__main__":
    demo.launch(share=True)