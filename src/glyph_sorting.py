import numpy as np

def sort_glyphs_advanced(detections):
    if not detections: return [], "rows"

    cartouches = []
    normal_glyphs = []

    v9_boxes = [d for d in detections if d[0] == 'V9']
    v9_claimed_indices = set()

    for v9 in v9_boxes:
        v9_idx = detections.index(v9)
        v9_claimed_indices.add(v9_idx)
        v9_x, v9_y, v9_w, v9_h = v9[1:]
        
        left, right = v9_x - (v9_w / 2), v9_x + (v9_w / 2)
        top, bottom = v9_y - (v9_h / 2), v9_y + (v9_h / 2)
        
        inside_glyphs = []
        for i, d in enumerate(detections):
            if i in v9_claimed_indices: continue
            gx, gy = d[1], d[2]
            if left <= gx <= right and top <= gy <= bottom:
                inside_glyphs.append(d)
                v9_claimed_indices.add(i)
        
        if inside_glyphs:
            inside_glyphs.sort(key=lambda x: x[2])
            cartouches.append(['V9'] + [g[0] for g in inside_glyphs])

    for i, d in enumerate(detections):
        if i not in v9_claimed_indices:
            normal_glyphs.append(d)

    if not normal_glyphs:
        return cartouches, "rows"

    avg_aspect = np.mean([d[3] / d[4] for d in normal_glyphs]) 
    layout = "columns" if avg_aspect < 0.8 else "rows"
    
    rows_or_cols = []

  
    if layout == "rows":
        normal_glyphs.sort(key=lambda d: d[2])
        median_h = np.median([d[4] for d in normal_glyphs])
        threshold = max(15, int(median_h * 0.8))

        current_group = [normal_glyphs[0]]
        for det in normal_glyphs[1:]:
            grp_y_avg = np.mean([d[2] for d in current_group])
            if abs(det[2] - grp_y_avg) <= threshold:
                current_group.append(det)
            else:
                rows_or_cols.append(current_group)
                current_group = [det]
        rows_or_cols.append(current_group)
        
        result = []
        for grp in rows_or_cols:
            grp_sorted = sorted(grp, key=lambda d: d[1], reverse=True) 
            result.append([g[0] for g in grp_sorted])
            
    else:
        normal_glyphs.sort(key=lambda d: d[1], reverse=True) 
        median_w = np.median([d[3] for d in normal_glyphs])
        threshold = max(15, int(median_w * 0.8))

        current_group = [normal_glyphs[0]]
        for det in normal_glyphs[1:]:
            grp_x_avg = np.mean([d[1] for d in current_group])
            if abs(det[1] - grp_x_avg) <= threshold:
                current_group.append(det)
            else:
                rows_or_cols.append(current_group)
                current_group = [det]
        rows_or_cols.append(current_group)
        
        result = []
        for grp in rows_or_cols:
            grp_sorted = sorted(grp, key=lambda d: d[2]) 
            result.append([g[0] for g in grp_sorted])

    return cartouches + result, layout