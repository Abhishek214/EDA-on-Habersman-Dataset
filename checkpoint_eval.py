#!/usr/bin/env python3

import os
import glob
import subprocess
import re
import json
from pathlib import Path

def extract_epoch_step(filename):
    """Extract epoch and step from checkpoint filename"""
    match = re.search(r'efficientdet-d(\d+)_(\d+)_(\d+)\.pth', filename)
    if match:
        compound_coef, epoch, step = map(int, match.groups())
        return compound_coef, epoch, step
    return None, None, None

def evaluate_checkpoint(weights_path, project='abhil', compound_coef=2):
    """Evaluate a single checkpoint and return mAP"""
    try:
        # Run evaluation
        cmd = [
            'python', 'coco_eval.py',
            '-p', project,
            '-c', str(compound_coef),
            '-w', weights_path,
            '--override', 'True'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        # Parse mAP from output
        output = result.stdout
        for line in output.split('\n'):
            if 'Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]' in line:
                map_score = float(line.split('=')[-1].strip())
                return map_score
                
    except Exception as e:
        print(f"Error evaluating {weights_path}: {e}")
        return None
    
    return None

def find_best_checkpoint(checkpoint_dir='logs/abhil/', project='abhil'):
    """Find the best checkpoint by evaluating all"""
    
    # Find all checkpoint files
    checkpoint_pattern = os.path.join(checkpoint_dir, 'efficientdet-d*.pth')
    checkpoints = glob.glob(checkpoint_pattern)
    
    if not checkpoints:
        print(f"No checkpoints found in {checkpoint_dir}")
        return None
    
    print(f"Found {len(checkpoints)} checkpoints")
    
    results = []
    
    for checkpoint in sorted(checkpoints):
        filename = os.path.basename(checkpoint)
        compound_coef, epoch, step = extract_epoch_step(filename)
        
        if compound_coef is None:
            continue
            
        print(f"Evaluating {filename}...")
        map_score = evaluate_checkpoint(checkpoint, project, compound_coef)
        
        if map_score is not None:
            results.append({
                'checkpoint': checkpoint,
                'filename': filename,
                'epoch': epoch,
                'step': step,
                'compound_coef': compound_coef,
                'mAP': map_score
            })
            print(f"  mAP: {map_score:.4f}")
        else:
            print(f"  Failed to evaluate")
    
    if not results:
        print("No successful evaluations")
        return None
    
    # Sort by mAP
    results.sort(key=lambda x: x['mAP'], reverse=True)
    
    # Print results
    print("\n" + "="*80)
    print("CHECKPOINT EVALUATION RESULTS")
    print("="*80)
    print(f"{'Rank':<5} {'Checkpoint':<40} {'Epoch':<6} {'Step':<8} {'mAP':<8}")
    print("-"*80)
    
    for i, result in enumerate(results[:10], 1):
        print(f"{i:<5} {result['filename']:<40} {result['epoch']:<6} {result['step']:<8} {result['mAP']:.4f}")
    
    # Save results
    with open('checkpoint_evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    best_checkpoint = results[0]
    print(f"\nBEST CHECKPOINT: {best_checkpoint['filename']}")
    print(f"Best mAP: {best_checkpoint['mAP']:.4f}")
    print(f"Epoch: {best_checkpoint['epoch']}, Step: {best_checkpoint['step']}")
    
    return best_checkpoint

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', default='logs/abhil/', help='Directory containing checkpoints')
    parser.add_argument('--project', default='abhil', help='Project name')
    args = parser.parse_args()
    
    best = find_best_checkpoint(args.checkpoint_dir, args.project)
