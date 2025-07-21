# Convert model to ONNX
python convert_onnx_fixed.py -w path/to/weights.pth -o model.onnx -c 2

# Evaluate ONNX model
python onnx_eval.py -p abhil -w model.onnx -c 2 --device cpu
