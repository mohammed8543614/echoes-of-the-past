import cv2
import re
from ultralytics import YOLO

from src.glyph_sorting import sort_glyphs_advanced
from src.translation import score_sequence, translate

MODEL_PATH = "best2.pt" 
try:
    yolo = YOLO(MODEL_PATH)
except Exception as e:
    print(f"⚠️ تأكد من وضع ملف الموديل في المجلد الصحيح: {e}")

VALID_GARDINER = re.compile(r'^[A-Z][0-9]+[a-zA-Z]?$')

def process_image(img_path_or_array):
    """
    تستقبل صورة، تمررها لـ YOLO، ثم تفرزها وتترجمها.
    ترجع: الصورة مع المربعات (للعرض)، والنص المترجم النهائي.
    """
    results = yolo(img_path_or_array)[0]
    
    annotated_img = results.plot()
    annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

    detections = []
    for box in results.boxes:
        if float(box.conf) < 0.25:
            continue
        code = results.names[int(box.cls)]
        if not VALID_GARDINER.match(code):
            continue
        x, y, w, h = map(float, box.xywh[0])
        detections.append((code, x, y, w, h))

    if not detections:
        return annotated_img, "❌ No hieroglyphs detected."

    blocks, layout = sort_glyphs_advanced(detections)

    if layout == "columns":
        flat_rtl = [code for block in blocks for code in block]
        flat_ltr = [code for block in reversed(blocks) for code in block]
    else:
        flat_rtl = [code for block in blocks for code in block]
        flat_ltr = [code for block in blocks for code in reversed(block)]
    
    score_rtl = score_sequence(flat_rtl)
    score_ltr = score_sequence(flat_ltr)
    rtl_better = score_rtl <= score_ltr
    
    translations = []
    for i, block in enumerate(blocks):
        if layout == "rows":
            codes = block if rtl_better else block[::-1]
        else:
            codes = block 
            
        trans = translate(codes)
        
        if 'V9' in codes:
            translations.append(f" Cartouche {i+1}: {trans}")
        else:
            translations.append(f" Block {i+1}: {trans}")

    final_translation = "\n".join(translations)
    
    return annotated_img, final_translation