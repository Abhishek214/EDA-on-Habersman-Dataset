import torch
import yaml
import argparse
import os
from backbone import EfficientDetBackbone

class Params:
    def __init__(self, project_file):
        self.params = yaml.safe_load(open(project_file).read())

    def __getattr__(self, item):
        return self.params.get(item, None)

def convert_to_onnx(project_file, weights_path, output_path, compound_coef=2):
    device = torch.device('cpu')
    params = Params(project_file)
    
    # Input sizes for different compound coefficients
    input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
    input_size = input_sizes[compound_coef]
    
    print(f"Loading model with {len(params.obj_list)} classes, compound_coef={compound_coef}")
    
    # Create model
    model = EfficientDetBackbone(
        num_classes=len(params.obj_list), 
        compound_coef=compound_coef, 
        onnx_export=True,
        ratios=eval(params.anchors_ratios), 
        scales=eval(params.anchors_scales)
    ).to(device)
    
    # Set swish to non-memory efficient for ONNX export
    model.backbone_net.model.set_swish(memory_efficient=False)
    
    # Load weights
    print(f"Loading weights from: {weights_path}")
    model.load_state_dict(torch.load(weights_path, map_location=device))
    
    # Set to evaluation mode
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn((1, 3, input_size, input_size), dtype=torch.float32).to(device)
    
    print(f"Converting to ONNX with input size: {input_size}x{input_size}")
    
    # Export to ONNX
    torch.onnx.export(
        model, 
        dummy_input, 
        output_path,
        verbose=True,
        input_names=['input'],
        output_names=['features', 'regression', 'classification', 'anchors'],
        opset_version=11,
        do_constant_folding=True,
        dynamic_axes={
            'input': {0: 'batch_size'},
            'regression': {0: 'batch_size'},
            'classification': {0: 'batch_size'},
            'anchors': {0: 'batch_size'}
        }
    )
    
    print(f"ONNX model saved to: {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Convert EfficientDet to ONNX')
    parser.add_argument('-p', '--project', type=str, default='projects/abhil.yml', help='Project file')
    parser.add_argument('-c', '--compound_coef', type=int, default=2, help='EfficientDet compound coefficient')
    parser.add_argument('-w', '--weights', type=str, required=True, help='Path to weights file')
    parser.add_argument('-o', '--output', type=str, default='efficientdet.onnx', help='Output ONNX file path')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.weights):
        raise FileNotFoundError(f"Weights file not found: {args.weights}")
    
    convert_to_onnx(args.project, args.weights, args.output, args.compound_coef)