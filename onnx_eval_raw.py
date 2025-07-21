import json
import os
import argparse
import numpy as np
import yaml
from tqdm import tqdm
import cv2

import onnxruntime as ort
import torch
from torch import nn

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess, boolean_string

class Params:
    def __init__(self, project_file):
        self.params = yaml.safe_load(open(project_file).read())

    def __getattr__(self, item):
        return self.params.get(item, None)

class ONNXEfficientDetEvaluator:
    def __init__(self, onnx_model_path, params, compound_coef=2, threshold=0.05, nms_threshold=0.5):
        """
        Initialize ONNX EfficientDet evaluator
        
        Args:
            onnx_model_path: Path to the ONNX model
            params: Parameters object with model configuration
            compound_coef: EfficientDet compound coefficient
            threshold: Confidence threshold for detections
            nms_threshold: NMS threshold
        """
        self.params = params
        self.compound_coef = compound_coef
        self.threshold = threshold
        self.nms_threshold = nms_threshold
        
        # Initialize ONNX Runtime session
        print(f"Loading ONNX model from {onnx_model_path}")
        self.ort_session = ort.InferenceSession(onnx_model_path)
        
        # Get input and output names
        self.input_name = self.ort_session.get_inputs()[0].name
        self.output_names = [output.name for output in self.ort_session.get_outputs()]
        print(f"Input name: {self.input_name}")
        print(f"Output names: {self.output_names}")
        
        # Initialize postprocessing components
        self.regressBoxes = BBoxTransform()
        self.clipBoxes = ClipBoxes()
        
        # Input sizes for different compound coefficients
        self.input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
        
    def predict(self, image_path):
        """
        Predict on a single image
        
        Args:
            image_path: Path to the image
            
        Returns:
            List of predictions in the format expected by COCO evaluation
        """
        # Preprocess image
        ori_imgs, framed_imgs, framed_metas = preprocess(
            image_path,
            max_size=self.input_sizes[self.compound_coef],
            mean=self.params.mean,
            std=self.params.std
        )
        
        # Convert to the format expected by ONNX (NCHW)
        input_data = framed_imgs[0].transpose(2, 0, 1)  # HWC to CHW
        input_data = np.expand_dims(input_data, axis=0)  # Add batch dimension
        input_data = input_data.astype(np.float32)
        
        # Run ONNX inference
        ort_inputs = {self.input_name: input_data}
        ort_outputs = self.ort_session.run(self.output_names, ort_inputs)
        
        # Parse outputs based on the model architecture
        # Expected outputs: features, regression, classification, anchors
        if len(ort_outputs) == 4:
            features, regression, classification, anchors = ort_outputs
        else:
            raise ValueError(f"Expected 4 outputs, got {len(ort_outputs)}")
        
        # Convert to PyTorch tensors for postprocessing
        regression = torch.from_numpy(regression)
        classification = torch.from_numpy(classification)
        anchors = torch.from_numpy(anchors)
        input_tensor = torch.from_numpy(input_data)
        
        # Apply postprocessing
        preds = self.postprocess_outputs(
            input_tensor, anchors, regression, classification
        )
        
        if not preds:
            return []
        
        # Invert affine transformation to get original image coordinates
        preds = invert_affine(framed_metas, preds)[0]
        
        return preds
        
    def postprocess_outputs(self, input_tensor, anchors, regression, classification):
        """
        Apply postprocessing to raw model outputs
        """
        # Transform regression deltas to bounding boxes
        transformed_anchors = self.regressBoxes(anchors, regression)
        
        # Clip boxes to image boundaries
        transformed_anchors = self.clipBoxes(transformed_anchors, input_tensor)
        
        # Apply confidence threshold and NMS
        scores = torch.max(classification, dim=2, keepdim=True)[0]
        scores_over_thresh = (scores > self.threshold)[0, :, 0]
        
        if scores_over_thresh.sum() == 0:
            return []
        
        classification = classification[:, scores_over_thresh, :]
        transformed_anchors = transformed_anchors[:, scores_over_thresh, :]
        scores = scores[:, scores_over_thresh, :]
        
        anchors_nms_idx = self.nms(
            torch.cat([transformed_anchors, scores], dim=2)[0, :, :], 
            self.nms_threshold
        )
        
        if anchors_nms_idx.shape[0] != 0:
            classes = torch.argmax(classification, dim=2)
            scores = torch.max(classification, dim=2)[0]
            
            boxes = transformed_anchors[0, anchors_nms_idx, :]
            classes = classes[0, anchors_nms_idx]
            scores = scores[0, anchors_nms_idx]
            
            return [{
                'rois': boxes.cpu().numpy(),
                'class_ids': classes.cpu().numpy(),
                'scores': scores.cpu().numpy()
            }]
        else:
            return []
    
    def nms(self, dets, thresh):
        """
        Non-Maximum Suppression
        """
        from torchvision.ops.boxes import nms as nms_torch
        return nms_torch(dets[:, :4], dets[:, 4], thresh)

def evaluate_coco_onnx(onnx_model_path, img_path, set_name, image_ids, coco, params, 
                       compound_coef=2, threshold=0.05, nms_threshold=0.5):
    """
    Evaluate ONNX model on COCO dataset
    """
    results = []
    
    # Initialize evaluator
    evaluator = ONNXEfficientDetEvaluator(
        onnx_model_path, params, compound_coef, threshold, nms_threshold
    )
    
    for image_id in tqdm(image_ids, desc="Evaluating images"):
        image_info = coco.loadImgs(image_id)[0]
        image_path = os.path.join(img_path, image_info['file_name'])
        
        # Skip if image doesn't exist
        if not os.path.exists(image_path):
            print(f"Warning: Image {image_path} not found")
            continue
        
        # Get predictions
        preds = evaluator.predict(image_path)
        
        if not preds:
            continue
            
        pred = preds[0]
        scores = pred['scores']
        class_ids = pred['class_ids']
        rois = pred['rois']
        
        if len(rois) > 0:
            # Convert [x1,y1,x2,y2] to [x1,y1,w,h] for COCO format
            rois_coco = rois.copy()
            rois_coco[:, 2] -= rois_coco[:, 0]  # width = x2 - x1
            rois_coco[:, 3] -= rois_coco[:, 1]  # height = y2 - y1
            
            for roi_id in range(len(rois_coco)):
                score = float(scores[roi_id])
                label = int(class_ids[roi_id])
                box = rois_coco[roi_id]
                
                # Skip very small boxes
                if box[2] < 1 or box[3] < 1:
                    continue
                
                image_result = {
                    'image_id': image_id,
                    'category_id': label + 1,  # COCO categories are 1-indexed
                    'score': score,
                    'bbox': box.tolist(),
                }
                
                results.append(image_result)
    
    if not results:
        raise Exception("The model does not provide any valid output, check model architecture and the data input")
    
    # Write output
    filepath = f'{set_name}_bbox_results_onnx.json'
    if os.path.exists(filepath):
        os.remove(filepath)
    json.dump(results, open(filepath, 'w'), indent=4)
    
    return filepath

def _eval(coco_gt, image_ids, pred_json_path):
    """
    Run COCO evaluation
    """
    # Load results in COCO evaluation tool
    coco_pred = coco_gt.loadRes(pred_json_path)
    
    # Run COCO evaluation
    print('BBox')
    coco_eval = COCOeval(coco_gt, coco_pred, 'bbox')
    coco_eval.params.imgIds = image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    return coco_eval

def main():
    parser = argparse.ArgumentParser('ONNX EfficientDet COCO Evaluation')
    parser.add_argument('-p', '--project', type=str, default='abhil', help='Project file that contains parameters')
    parser.add_argument('-c', '--compound_coef', type=int, default=2, help='Coefficients of efficientdet')
    parser.add_argument('-w', '--weights', type=str, default='efficientdet-d2.onnx', help='Path to ONNX weights')
    parser.add_argument('--threshold', type=float, default=0.05, help='Confidence threshold')
    parser.add_argument('--nms_threshold', type=float, default=0.5, help='NMS threshold')
    parser.add_argument('--override', type=boolean_string, default=True, help='Override previous bbox results file if exists')
    parser.add_argument('--max_images', type=int, default=100000, help='Maximum number of images to evaluate')
    
    args = parser.parse_args()
    
    # Load parameters
    params = Params(f'projects/{args.project}.yml')
    
    # Dataset paths
    SET_NAME = params.val_set
    VAL_GT = f'datasets/{params.project_name}/annotations/instances_{SET_NAME}.json'
    VAL_IMGS = f'datasets/{params.project_name}/{SET_NAME}/'
    
    print(f'Running ONNX COCO-style evaluation on project {args.project}')
    print(f'ONNX model: {args.weights}')
    print(f'Dataset: {VAL_GT}')
    print(f'Images: {VAL_IMGS}')
    
    # Load COCO dataset
    coco_gt = COCO(VAL_GT)
    image_ids = coco_gt.getImgIds()[:args.max_images]
    
    print(f'Number of images to evaluate: {len(image_ids)}')
    
    results_file = f'{SET_NAME}_bbox_results_onnx.json'
    
    if args.override or not os.path.exists(results_file):
        print("Running evaluation...")
        results_file = evaluate_coco_onnx(
            args.weights, VAL_IMGS, SET_NAME, image_ids, coco_gt, params,
            args.compound_coef, args.threshold, args.nms_threshold
        )
        print(f"Results saved to: {results_file}")
    else:
        print(f"Using existing results file: {results_file}")
    
    # Run COCO evaluation
    print("\nRunning COCO evaluation...")
    coco_eval = _eval(coco_gt, image_ids, results_file)
    
    # Print summary
    print("\nEvaluation completed!")
    print(f"mAP: {coco_eval.stats[0]:.4f}")
    print(f"mAP@0.5: {coco_eval.stats[1]:.4f}")
    print(f"mAP@0.75: {coco_eval.stats[2]:.4f}")

if __name__ == '__main__':
    main()
