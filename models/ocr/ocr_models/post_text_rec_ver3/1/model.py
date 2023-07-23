import io
import json
import os
import sys
from pathlib import Path

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
            self.model_config, "TEXTS_RECS"
        )
        self.output0_dtype = pb_utils.triton_string_to_numpy(
            output0_config["data_type"]
        )

        output1_config = pb_utils.get_output_config_by_name(
            self.model_config, "SCORES_RECS"
        )
        self.output1_dtype = pb_utils.triton_string_to_numpy(
            output1_config["data_type"]
        )

        path_cur = os.path.dirname(os.path.abspath(__file__))
        
        postprocess_params = {
            "name": "CTCLabelDecode",
            "character_dict_path": os.path.join(
                path_cur, "ppocr/utils/dict/japan_dict.txt"
            ),
            "use_space_char": True,
        }
        # model_file_path="weights/onnx/onnx_japan_mobile_v2.0_rec/model.onnx"
        self.postprocess_op = build_post_process(postprocess_params)
        self.max_text_length = 20
        self.dictionary = (
            open(os.path.join(path_cur, "ppocr/utils/dict/japan_dict.txt"), "r", encoding='utf-8')
            .read()
            .split("\n")
        )

        self.dectionary_map = {x: i for i, x in enumerate(self.dictionary)}

    def encode(self, text):
        text_encode = [-1] * self.max_text_length
        for i, x in enumerate(text):
            text_encode[i] = self.dectionary_map[x]
            if i == self.max_text_length - 1:
                break
        return np.array(text_encode)

    def execute(self, requests):
        responses = []

        for request in requests:
            # Get INPUT0
            preds = pb_utils.get_input_tensor_by_name(
                request, "OUTPUT_INFER_RECS"
            ).as_numpy()

            indexs = pb_utils.get_input_tensor_by_name(
                request, "INDEXS"
            ).as_numpy()

            rec_result = self.postprocess_op(preds)
            text_recs = np.array([x[0] for x in rec_result])

            # Custom metadata
            # text_rec_array = np.array(text_recs[0], dtype=np.object)
            # custom_output_tensor = pb_utils.Tensor("CUSTOM_TEXT_OUTPUT", text_rec_array)
            # custom_output_tensors = [custom_output_tensor]
            # custom_response = pb_utils.InferenceResponse(output_tensors=output_tensors)
            # responses.append(custom_response)

            text_recs_expanded = []

            for i in range(len(indexs)):
                if indexs[i] == [1]:
                    text_recs_expanded.append(text_recs[i])
                else:
                    text_recs_expanded.append("")

            text_recs_np = []
            for text_rec in text_recs_expanded:
                text_recs_np.append(self.encode(text_rec))

            text_recs_np = np.array(text_recs_np)
            
            scores_rec_np = []
            scores = np.array([float(x[1]) for x in rec_result])
            j = 0
            for i in range(len(indexs)):
                if indexs[i] == [1]:
                    scores_rec_np.append(scores[i])
                else:
                    scores_rec_np.append(0)
            scores_rec_np = np.array(scores_rec_np)
            text_recs_np = np.ascontiguousarray(text_recs_np, dtype=np.int32)
            # print("[Debug] {}".format(text_recs_np.shape))
            out_tensor_0 = pb_utils.Tensor(
                "TEXTS_RECS", text_recs_np.astype(self.output0_dtype)
            )
            out_tensor_1 = pb_utils.Tensor(
                "SCORES_RECS", scores_rec_np.astype(self.output1_dtype)
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
