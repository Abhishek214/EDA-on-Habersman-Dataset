import torch
import torch.nn as nn
import yaml
import argparse
import os
import onnx
import onnx_graphsurgeon
import numpy as np
from collections import OrderedDict
from backbone import EfficientDetBackbone
from efficientdet.utils import BBoxTransform, ClipBoxes

def efficientdet_insert_nms(path, score_threshold=0.05, iou_threshold=0.5, max_output_boxes=100):
    """Insert NMS into EfficientDet ONNX model"""
    onnx_model = onnx.load(path)
    graph = onnx_graphsurgeon.import_onnx(onnx_model)
    graph.toposort()
    graph.fold_constants()
    graph.cleanup()

    topk = max_output_boxes
    attrs = OrderedDict(
        plugin_version='1',
        background_class=-1,
        max_output_boxes=topk,
        score_threshold=score_threshold,
        iou_threshold=iou_threshold,
        score_activation=False,
        box_coding=0,
    )

    outputs = [
        onnx_graphsurgeon.Variable('num_dets', np.int32, [-1, 1]),
        onnx_graphsurgeon.Variable('det_boxes', np.float32, [-1, topk, 4]),
        onnx_graphsurgeon.Variable('det_scores', np.float32, [-1, topk]),
        onnx_graphsurgeon.Variable('det_classes', np.int32, [-1, topk])
    ]

    graph.layer(
        op='EfficientNMS_TRT', 
        name="batched_nms", 
        inputs=[graph.outputs[0], graph.outputs[1]], 
        outputs=outputs, 
        attrs=attrs,
    )

    graph.outputs = outputs
    graph.cleanup().toposort()
    
    nms_path = path.replace('.onnx', '_with_nms.onnx')
    onnx.save(onnx_graphsurgeon.export_onnx(graph), nms_path)
    return nms_path

class EfficientDetWithPostProcess(nn.Module):
    def __init__(self, backbone, score_threshold=0.05):
        super().__init__()
        self.backbone = backbone
        self.score_threshold = score_threshold
        self.regressBoxes = BBoxTransform()
        self.clipBoxes = ClipBoxes()
    
    def forward(self, x):
        features, regression, classification, anchors = self.backbone(x)
        
        # Transform boxes
        transformed_anchors = self.regressBoxes(anchors, regression)
        transformed_anchors = self.clipBoxes(transformed_anchors, x)
        
        # Get classification scores
        classification_scores = torch.sigmoid(classification)
        
        # Flatten for NMS input - [batch, num_boxes, 4] and [batch, num_boxes, num_classes]
        boxes = transformed_anchors.squeeze(0)  # Remove batch dim: [num_boxes, 4]
        scores = classification_scores.squeeze(0)  # Remove batch dim: [num_boxes, num_classes]
        
        return boxes, scores


class Params:
    def __init__(self, project_file):
        self.params = yaml.safe_load(open(project_file).read())

    def __getattr__(self, item):
        return self.params.get(item, None)

def convert_to_onnx(project_file, weights_path, output_path, compound_coef=2, score_threshold=0.05, nms_threshold=0.5, max_detections=100):
    device = torch.device('cpu')
    params = Params(project_file)
    
    # Input sizes for different compound coefficients
    input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
    input_size = input_sizes[compound_coef]
    
    print(f"Loading model with {len(params.obj_list)} classes, compound_coef={compound_coef}")
    
    # Create backbone model
    backbone = EfficientDetBackbone(
        num_classes=len(params.obj_list), 
        compound_coef=compound_coef, 
        onnx_export=True,
        ratios=eval(params.anchors_ratios), 
        scales=eval(params.anchors_scales)
    ).to(device)
    
    # Set swish to non-memory efficient for ONNX export
    backbone.backbone_net.model.set_swish(memory_efficient=False)
    
    # Load weights
    print(f"Loading weights from: {weights_path}")
    backbone.load_state_dict(torch.load(weights_path, map_location=device))
    
    # Create model with postprocessing (no NMS yet)
    model = EfficientDetWithPostProcess(backbone, score_threshold)
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn((1, 3, input_size, input_size), dtype=torch.float32).to(device)
    
    print(f"Converting to ONNX (score_thresh={score_threshold})")
    
    # Export to ONNX without NMS first
    torch.onnx.export(
        model, 
        dummy_input, 
        output_path,
        verbose=True,
        input_names=['input'],
        output_names=['boxes', 'scores'],
        opset_version=11,
        do_constant_folding=True,
        dynamic_axes={
            'input': {0: 'batch_size'},
            'boxes': {0: 'num_anchors'},
            'scores': {0: 'num_anchors'}
        }
    )
    
    print(f"ONNX model saved to: {output_path}")
    
    # Insert NMS into the ONNX model
    print("Inserting NMS into ONNX model...")
    nms_path = efficientdet_insert_nms(output_path, score_threshold, nms_threshold, max_detections)
    print(f"ONNX model with NMS saved to: {nms_path}")
    
    return nms_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Convert EfficientDet to ONNX with NMS')
    parser.add_argument('-p', '--project', type=str, default='projects/abhil.yml', help='Project file')
    parser.add_argument('-c', '--compound_coef', type=int, default=2, help='EfficientDet compound coefficient')
    parser.add_argument('-w', '--weights', type=str, required=True, help='Path to weights file')
    parser.add_argument('-o', '--output', type=str, default='efficientdet.onnx', help='Output ONNX file path')
    parser.add_argument('--score_threshold', type=float, default=0.05, help='Score threshold for filtering detections')
    parser.add_argument('--nms_threshold', type=float, default=0.5, help='NMS threshold')
    parser.add_argument('--max_detections', type=int, default=100, help='Maximum number of detections to output')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.weights):
        raise FileNotFoundError(f"Weights file not found: {args.weights}")
    
    nms_model_path = convert_to_onnx(args.project, args.weights, args.output, args.compound_coef, 
                                    args.score_threshold, args.nms_threshold, args.max_detections)
    print(f"Final model with NMS: {nms_model_path}")