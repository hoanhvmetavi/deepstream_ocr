import io
import json
import os
import sys

import cv2
import numpy as np
import triton_python_backend_utils as pb_utils
from ppocr.data import create_operators, transform
from ppocr.postprocess import build_post_process


class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        self.model_config = json.loads(args["model_config"])

        # Get OUTPUT0 configuration
        output0_config = pb_utils.get_output_config_by_name(
            self.model_config, "INPUT_DETS"
        )

        output1_config = pb_utils.get_output_config_by_name(
            self.model_config, "SHAPE_LISTS"
        )

        self.output0_dtype = pb_utils.triton_string_to_numpy(
            output0_config["data_type"]
        )

        self.output1_config = pb_utils.triton_string_to_numpy(
            output1_config["data_type"]
        )

        pre_process_list = [
            {
                "DetResizeForTest": {
                    "limit_side_len": 960,
                    "limit_type": "max",
                }
            },
            {
                "NormalizeImage": {
                    "std": [0.229, 0.224, 0.225],
                    "mean": [0.485, 0.456, 0.406],
                    "scale": "1./255.",
                    "order": "hwc",
                }
            },
            {"ToCHWImage": None},
            {"KeepKeys": {"keep_keys": ["image", "shape"]}},
        ]
        img_h, img_w = None, None

        if img_h is not None and img_w is not None and img_h > 0 and img_w > 0:
            pre_process_list[0] = {
                "DetResizeForTest": {"image_shape": [img_h, img_w]}
            }
        self.preprocess_op = create_operators(pre_process_list)

    def execute(self, requests):
        output0_dtype = self.output0_dtype
        responses = []

        for request in requests:
            # Get INPUT0
            imgs = pb_utils.get_input_tensor_by_name(
                request, "IMAGES"
            ).as_numpy()
            imgs = np.transpose(imgs, (0, 2, 3, 1))
            datas = []
            shape_lists = []
            for i, img in enumerate(imgs):
                data = {"image": img}
                data = transform(data, self.preprocess_op)
                img, shape_list = data
                # img = np.expand_dims(img, axis=0)
                datas.append(img)
                shape_lists.append(shape_list)
            shape_lists = np.array(shape_lists)
            datas = np.array(datas)
            out_0_np = np.ascontiguousarray(datas, dtype=np.int8)
            out_tensor_0 = pb_utils.Tensor(
                "INPUT_DETS", out_0_np.astype(output0_dtype)
            )

            out_tensor_1 = pb_utils.Tensor(
                "SHAPE_LISTS", shape_lists.astype(output0_dtype)
            )
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[out_tensor_0, out_tensor_1]
            )
            responses.append(inference_response)

        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is OPTIONAL. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print("Cleaning up...")
