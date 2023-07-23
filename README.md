# DEEPSTREAM SIMPLE PIPELINE FOR TEST OCR JAPANESE MODEL.

Pipeline: source > streammux > pgie (Human detection) > tracker > nvinferserver (PaddleOCR Ensemble) > tiler > nvvidconv > nvosd > sink.

## Usage
- Run the main deepstream code with python:
python3 main.py

## Model usage
- pgie: Can change pgie model to any human detection model.

- nvinferserver: This is an emsemble model of two step: Text detection and Text recognition. The model are stored in models/ocr/ocr_models

When running OCR model first time on any new PC, need to reconvert two infer model using the scripts: models/ocr/ocr_models/infer_text_rec_ver3/1/convert.sh and models/ocr/ocr_models/infer_text_det_ver3/1/convert.sh

Also, you need to build custom parser so file to run Paddle model. Build in models/ocr/custom_parser. run CUDA_VER=11.6 make

