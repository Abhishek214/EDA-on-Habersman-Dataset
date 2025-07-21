import torch
import torch.nn as nn
import yaml
import argparse
import os
from backbone import EfficientDetBackbone
from efficientdet.utils import BBoxTransform, ClipBoxes
from torchvision.ops import nms

class EfficientDetWithPostProcess(nn.Module):
    def __init__(self, backbone, score_threshold=0.05, nms_threshold=0.5):
        super().__init__()
        self.backbone = backbone
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.regressBoxes = BBoxTransform()
        self.clipBoxes = ClipBoxes()
    
    def forward(self, x):
        features, regression, classification, anchors = self.backbone(x)
        
        # Transform boxes
        transformed_anchors = self.regressBoxes(anchors, regression)
        transformed_anchors = self.clipBoxes(transformed_anchors, x)
        
        # Get scores and filter by threshold
        scores = torch.max(classification, dim=2, keepdim=True)[0]
        scores_over_thresh = (scores > self.score_threshold)[0, :, 0]
        
        if scores_over_thresh.sum() == 0:
            # Return empty results
            return (torch.zeros((0, 4)), torch.zeros((0,)), torch.zeros((0,), dtype=torch.long))
        
        classification = classification[:, scores_over_thresh, :]
        transformed_anchors = transformed_anchors[:, scores_over_thresh, :]
        scores = scores[:, scores_over_thresh, :]
        
        anchors_nms_idx = nms(transformed_anchors[0,:,:], scores[0,:,0], self.nms_threshold)
        
        if anchors_nms_idx.shape[0] != 0:
            classes = torch.argmax(classification, dim=2)
            
            # Final results
            boxes = transformed_anchors[0, anchors_nms_idx, :]
            scores_final = scores[0, anchors_nms_idx, 0] 
            classes_final = classes[0, anchors_nms_idx]
            
            return boxes, scores_final, classes_final
        else:
            return (torch.zeros((0, 4)), torch.zeros((0,)), torch.zeros((0,), dtype=torch.long))


class Params:
    def __init__(self, project_file):
        self.params = yaml.safe_load(open(project_file).read())

    def __getattr__(self, item):
        return self.params.get(item, None)

def convert_to_onnx(project_file, weights_path, output_path, compound_coef=2, score_threshold=0.05, nms_threshold=0.5):
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
    
    # Create model with postprocessing
    model = EfficientDetWithPostProcess(backbone, score_threshold, nms_threshold)
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn((1, 3, input_size, input_size), dtype=torch.float32).to(device)
    
    print(f"Converting to ONNX with postprocessing (score_thresh={score_threshold}, nms_thresh={nms_threshold})")
    
    # Export to ONNX
    torch.onnx.export(
        model, 
        dummy_input, 
        output_path,
        verbose=True,
        input_names=['input'],
        output_names=['boxes', 'scores', 'classes'],
        opset_version=11,
        do_constant_folding=True,
        dynamic_axes={
            'input': {0: 'batch_size'},
            'boxes': {0: 'num_detections'},
            'scores': {0: 'num_detections'},
            'classes': {0: 'num_detections'}
        }
    )
    
    print(f"ONNX model with postprocessing saved to: {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Convert EfficientDet to ONNX with postprocessing')
    parser.add_argument('-p', '--project', type=str, default='projects/abhil.yml', help='Project file')
    parser.add_argument('-c', '--compound_coef', type=int, default=2, help='EfficientDet compound coefficient')
    parser.add_argument('-w', '--weights', type=str, required=True, help='Path to weights file')
    parser.add_argument('-o', '--output', type=str, default='efficientdet.onnx', help='Output ONNX file path')
    parser.add_argument('--score_threshold', type=float, default=0.05, help='Score threshold for filtering detections')
    parser.add_argument('--nms_threshold', type=float, default=0.5, help='NMS threshold')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.weights):
        raise FileNotFoundError(f"Weights file not found: {args.weights}")
    
    convert_to_onnx(args.project, args.weights, args.output, args.compound_coef, 
                   args.score_threshold, args.nms_threshold)