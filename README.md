scale = min(input_size / orig_w, input_size / orig_h)
    new_w = int(orig_w * scale)  # 768
    new_h = int(orig_h * scale)  # 542
    
    # Calculate padding
    padding_w = (input_size - new_w) // 2  # 0
    padding_h = (input_size - new_h) // 2  # 113
    
    # Convert boxes
    boxes[:, [0, 2]] -= padding_w  # Remove horizontal padding
    boxes[:, [1, 3]] -= padding_h  # Remove vertical padding
    boxes /= scale  # Scale back to original size
    
    # Clip to original image bounds
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h)
    
    return boxes
