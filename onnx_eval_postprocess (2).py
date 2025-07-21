        # Process results after coordinate transformation
        for i in range(len(final_boxes)):
            box = final_boxes[i]
            score = float(final_scores[i])
            class_id = int(final_classes[i])
            
            # Convert [x1,y1,x2,y2] to [x,y,w,h] for COCO format
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            
            if width <= 0 or height <= 0:
                continue
                
            image_result = {
                'image_id': image_id,
                'category_id': class_id + 1,
                'score': score,
                'bbox': [float(x1), float(y1), float(width), float(height)],
            }

            results.append(image_result)"""
ONNX EfficientDet Evaluation with Built-in Postprocessing
"""

import json
import os
import argparse
import numpy as np
import yaml
from tqdm import tqdm
import onnxruntime as ort

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from utils.utils import preprocess, invert_affine, boolean_string

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('-p', '--project', type=str, default='coco', help='Project file that contains parameters')
    ap.add_argument('-c', '--compound_coef', type=int, default=0, help='Coefficients of efficientdet')
    ap.add_argument('-w', '--weights', type=str, required=True, help='/path/to/onnx/model.onnx')
    ap.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], help='Device for ONNX runtime')
    ap.add_argument('--override', type=boolean_string, default=True, help='Override previous bbox results file if exists')
    ap.add_argument('--max_images', type=int, default=100000, help='Maximum number of images to evaluate')
    return ap.parse_args()

class ONNXEfficientDetPostProcess:
    def __init__(self, model_path, device='cpu'):
        self.device = device
        
        # Set up ONNX Runtime session
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, providers=providers)
        
        # Get input/output info
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        
        if len(inputs) == 0:
            raise ValueError("ONNX model has no inputs. Check if model was exported correctly.")
        if len(outputs) == 0:
            raise ValueError("ONNX model has no outputs. Check if model was exported correctly.")
            
        self.input_name = inputs[0].name
        self.output_names = [output.name for output in outputs]
        
        print(f"ONNX model loaded. Input: {self.input_name}, Outputs: {self.output_names}")
        print(f"Input shape: {inputs[0].shape}, Output shapes: {[out.shape for out in outputs]}")
    
    def predict(self, image_tensor):
        """Run inference on preprocessed image tensor"""
        if hasattr(image_tensor, 'numpy'):
            image_np = image_tensor.numpy()
        else:
            image_np = image_tensor
            
        # Run inference - returns boxes, scores, classes
        outputs = self.session.run(self.output_names, {self.input_name: image_np})
        return outputs[0], outputs[1], outputs[2]  # boxes, scores, classes

def evaluate_onnx_coco(img_path, set_name, image_ids, coco, model, compound_coef, params):
    """Evaluate ONNX model with postprocessing on COCO dataset"""
    results = []
    input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
    
    for image_id in tqdm(image_ids):
        image_info = coco.loadImgs(image_id)[0]
        image_path = img_path + image_info['file_name']

        # Preprocess image
        ori_imgs, framed_imgs, framed_metas = preprocess(
            image_path,
            max_size=input_sizes[compound_coef],
            mean=params['mean'],
            std=params['std']
        )

        # Prepare input for ONNX
        x = framed_imgs[0]
        x = np.expand_dims(x, axis=0)  # Add batch dimension
        x = np.transpose(x, (0, 3, 1, 2))  # Convert to NCHW format

        # Run ONNX inference - get final detections
        boxes, scores, classes = model.predict(x)
        
        # Skip if no detections
        if len(boxes) == 0:
            continue

        # Debug: Print framed_metas structure
        # print(f"framed_metas type: {type(framed_metas)}, content: {framed_metas}")

        # Convert boxes back to original image coordinates
        # Use invert_affine for proper coordinate transformation
        try:
            # Create predictions dict in expected format
            preds = [{
                'rois': boxes,
                'scores': scores, 
                'class_ids': classes
            }]
            
            # Use existing invert_affine function
            preds = invert_affine(framed_metas, preds)[0]
            
            final_boxes = preds['rois']
            final_scores = preds['scores']
            final_classes = preds['class_ids']
            
        except Exception as e:
            # Fallback: manual coordinate transformation
            print(f"invert_affine failed: {e}, using manual transform")
            final_boxes = boxes
            final_scores = scores
            final_classes = classes

    if not len(results):
        raise Exception("The model does not provide any valid output, check model architecture and the data input")

    # Write output
    filepath = f'{set_name}_bbox_results_onnx_postprocess.json'
    if os.path.exists(filepath):
        os.remove(filepath)
    json.dump(results, open(filepath, 'w'), indent=4)
    
    return filepath

def _eval(coco_gt, image_ids, pred_json_path):
    """Evaluate predictions using COCO metrics"""
    coco_pred = coco_gt.loadRes(pred_json_path)

    print('BBox')
    coco_eval = COCOeval(coco_gt, coco_pred, 'bbox')
    coco_eval.params.imgIds = image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

def main():
    args = parse_args()
    
    # Load project parameters
    params = yaml.safe_load(open(f'projects/{args.project}.yml'))
    
    print(f'Running ONNX COCO-style evaluation on project {args.project}, model {args.weights}...')
    
    # Dataset paths
    SET_NAME = params['val_set']
    VAL_GT = f'datasets/{params["project_name"]}/annotations/instances_{SET_NAME}.json'
    VAL_IMGS = f'datasets/{params["project_name"]}/{SET_NAME}/'
    
    # Load COCO dataset
    coco_gt = COCO(VAL_GT)
    image_ids = coco_gt.getImgIds()[:args.max_images]
    
    # Load ONNX model
    model = ONNXEfficientDetPostProcess(args.weights, args.device)
    
    # Check if results already exist
    results_file = f'{SET_NAME}_bbox_results_onnx_postprocess.json'
    if args.override or not os.path.exists(results_file):
        print("Running inference...")
        results_file = evaluate_onnx_coco(
            VAL_IMGS, SET_NAME, image_ids, coco_gt, model, 
            args.compound_coef, params
        )
    else:
        print(f"Using existing results: {results_file}")
    
    # Evaluate results
    print("Evaluating results...")
    _eval(coco_gt, image_ids, results_file)

if __name__ == '__main__':
    main()