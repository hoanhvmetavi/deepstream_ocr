/usr/src/tensorrt/bin/trtexec  --explicitBatch --onnx=model.onnx --minShapes=x:1x3x48x100 --optShapes=x:16x3x48x100 --maxShapes=x:32x3x48x100 --saveEngine=model.plan  --device=0 --verbose
