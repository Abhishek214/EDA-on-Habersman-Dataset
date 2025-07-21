"""
ONNX EfficientDet COCO-Style Evaluation

Put images here: datasets/your_project_name/val_set_name/*.jpg  
Put annotations here: datasets/your_project_name/annotations/instances_{val_set_name}.json  
Put ONNX model here: /path/to/your/model.onnx
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

from utils.utils import preprocess, invert_affine, postprocess, boolean_string

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('-p', '--project', type=str, default='coco', help='Project file that contains parameters')
    ap.add_argument('-c', '--compound_coef', type=int, default=0, help='Coefficients of efficientdet')
    ap.add_argument('-w', '--weights', type=str, required=True, help='/path/to/onnx/model.onnx')
    ap.add_argument('--nms_threshold', type=float, default=0.5, help='NMS threshold')
    ap.add_argument('--score_threshold', type=float, default=0.05, help='Score threshold for detections')
    ap.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], help='Device for ONNX runtime')
    ap.add_argument('--override', type=boolean_string, default=True, help='Override previous bbox results file if exists')
    ap.add_argument('--max_images', type=int, default=100000, help='Maximum number of images to evaluate')
    return ap.parse_args()

class ONNXEfficientDet:
    def __init__(self, model_path, device='cpu'):
        self.device = device
        
        # Set up ONNX Runtime session
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, providers=providers)
        
        # Get input/output info
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        
        print(f"ONNX model loaded. Input: {self.input_name}, Outputs: {self.output_names}")
    
    def predict(self, image_tensor):
        """Run inference on preprocessed image tensor"""
        # Convert to numpy if needed
        if hasattr(image_tensor, 'numpy'):
            image_np = image_tensor.numpy()
        else:
            image_np = image_tensor
            
        # Run inference
        outputs = self.session.run(self.output_names, {self.input_name: image_np})
        
        # Return in same format as PyTorch model: features, regression, classification, anchors
        return outputs

def evaluate_onnx_coco(img_path, set_name, image_ids, coco, model, compound_coef, params, threshold=0.05, nms_threshold=0.5):
    """Evaluate ONNX model on COCO dataset"""
    results = []
    input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
    
    for image_id in tqdm(image_ids):
        image_info = coco.loadImgs(image_id)[0]
        image_path = img_path + image_info['file_name']

        # Preprocess image (same as PyTorch version)
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

        # Run ONNX inference
        outputs = model.predict(x)
        
        # Handle 8 outputs: features are split into pyramid levels
        if len(outputs) == 8:
            # Outputs: [P3, P4, P5, P6, P7, regression, classification, anchors]
            features = outputs[:5]  # P3-P7 feature maps
            regression = outputs[5]
            classification = outputs[6] 
            anchors = outputs[7]
        else:
            # Fallback for 4 outputs
            features, regression, classification, anchors = outputs

        # Convert outputs to expected format for postprocessing
        # Note: Postprocessing expects torch tensors, so we might need to adapt
        # For now, assuming postprocess function can handle numpy arrays or we convert back
        try:
            import torch
            # Convert back to torch tensors for postprocessing compatibility
            regression_tensor = torch.from_numpy(regression)
            classification_tensor = torch.from_numpy(classification)
            anchors_tensor = torch.from_numpy(anchors)
            x_tensor = torch.from_numpy(x)
            
            # Create BBoxTransform and ClipBoxes (same as original)
            from efficientdet.utils import BBoxTransform, ClipBoxes
            regressBoxes = BBoxTransform()
            clipBoxes = ClipBoxes()
            
            preds = postprocess(
                x_tensor,
                anchors_tensor, regression_tensor, classification_tensor,
                regressBoxes, clipBoxes,
                threshold, nms_threshold
            )
        except Exception as e:
            print(f"Error in postprocessing: {e}")
            continue

        if not preds:
            continue

        preds = invert_affine(framed_metas, preds)[0]

        scores = preds['scores']
        class_ids = preds['class_ids']
        rois = preds['rois']

        if rois.shape[0] > 0:
            # Convert [x1,y1,x2,y2] to [x1,y1,w,h]
            rois[:, 2] -= rois[:, 0]
            rois[:, 3] -= rois[:, 1]

            bbox_score = scores

            for roi_id in range(rois.shape[0]):
                score = float(bbox_score[roi_id])
                label = int(class_ids[roi_id])
                box = rois[roi_id, :]

                image_result = {
                    'image_id': image_id,
                    'category_id': label + 1,
                    'score': float(score),
                    'bbox': box.tolist(),
                }

                results.append(image_result)

    if not len(results):
        raise Exception("The model does not provide any valid output, check model architecture and the data input")

    # Write output
    filepath = f'{set_name}_bbox_results_onnx.json'
    if os.path.exists(filepath):
        os.remove(filepath)
    json.dump(results, open(filepath, 'w'), indent=4)
    
    return filepath

def _eval(coco_gt, image_ids, pred_json_path):
    """Evaluate predictions using COCO metrics"""
    # Load results in COCO evaluation tool
    coco_pred = coco_gt.loadRes(pred_json_path)

    # Run COCO evaluation
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
    model = ONNXEfficientDet(args.weights, args.device)
    
    # Check if results already exist
    results_file = f'{SET_NAME}_bbox_results_onnx.json'
    if args.override or not os.path.exists(results_file):
        print("Running inference...")
        results_file = evaluate_onnx_coco(
            VAL_IMGS, SET_NAME, image_ids, coco_gt, model, 
            args.compound_coef, params, args.score_threshold, args.nms_threshold
        )
    else:
        print(f"Using existing results: {results_file}")
    
    # Evaluate results
    print("Evaluating results...")
    _eval(coco_gt, image_ids, results_file)

if __name__ == '__main__':
    main()