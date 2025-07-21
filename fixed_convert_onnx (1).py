import torch
import yaml
from torch import nn
from backbone import EfficientDetBackbone
import numpy as np

class Params:
    def __init__(self, project_file):
        self.params = yaml.safe_load(open(project_file).read())

    def __getattr__(self, item):
        return self.params.get(item, None)

device = torch.device('cpu')
params = Params('projects/abhil.yml')

# Load model
model = EfficientDetBackbone(
    num_classes=len(params.obj_list), 
    compound_coef=2, 
    onnx_export=True,
    ratios=eval(params.anchors_ratios), 
    scales=eval(params.anchors_scales)
).to(device)

model.backbone_net.model.set_swish(memory_efficient=False)

dummy_input = torch.randn((1, 3, 768, 768), dtype=torch.float32).to(device)

# Load weights
model.load_state_dict(torch.load('F:/opt/disk1/app/Abhishek/Yet-Another-EfficientDet-Pytorch-master/logs/abhil/efficientdet-d2_49_5784.pth'))

# Set to eval mode
model.eval()

# Test the model output first
with torch.no_grad():
    test_output = model(dummy_input)
    print(f"Model outputs: {len(test_output)}")
    for i, out in enumerate(test_output):
        print(f"Output {i} shape: {out.shape}")

# Export with proper output names
print("Exporting ONNX model...")
torch.onnx.export(
    model, 
    dummy_input, 
    "efficientdet-d2-fixed.onnx", 
    verbose=True, 
    input_names=['input'], 
    output_names=['features', 'regression', 'classification', 'anchors'],
    opset_version=11,
    do_constant_folding=True,
    dynamic_axes={
        'input': {0: 'batch_size'},
        'features': {0: 'batch_size'},
        'regression': {0: 'batch_size'}, 
        'classification': {0: 'batch_size'},
        'anchors': {0: 'batch_size'}
    }
)

print("ONNX export completed: efficientdet-d2-fixed.onnx")
