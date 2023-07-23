import json

import numpy as np
import triton_python_backend_utils as pb_utils
from ppocr.postprocess import build_post_process


class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        self.model_config = json.loads(args["model_config"])

        # Get OUTPUT0 configuration
        output0_config = pb_utils.get_output_config_by_name(
            self.model_config, "OUTPUT_DETS"
        )
        self.output0_dtype = pb_utils.triton_string_to_numpy(
            output0_config["data_type"]
        )
        postprocess_params = {}
        postprocess_params["name"] = "DBPostProcess"
        postprocess_params["thresh"] = 0.45
        postprocess_params["box_thresh"] = 0.5
        postprocess_params["max_candidates"] = 100
        postprocess_params["unclip_ratio"] = 1.5
        postprocess_params["use_dilation"] = False
        postprocess_params["score_mode"] = "fast"
        self.min_width = 15
        self.min_height = 15
        self.max_box = 1
        self.postprocess_op = build_post_process(postprocess_params)

    def order_points_clockwise(self, pts):
        xSorted = pts[np.argsort(pts[:, 0]), :]
        leftMost = xSorted[:2, :]
        rightMost = xSorted[2:, :]
        leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
        (tl, bl) = leftMost
        rightMost = rightMost[np.argsort(rightMost[:, 1]), :]
        (tr, br) = rightMost
        rect = np.array([tl, tr, br, bl], dtype="float32")
        return rect

    def filter_tag_det_res(self, dt_boxes, dt_scores, image_shape):
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        dt_scores_new = []
        for box, score in zip(dt_boxes, dt_scores):
            box = self.order_points_clockwise(box)
            box = self.clip_det_res(box, img_height, img_width)
            rect_width = int(np.linalg.norm(box[0] - box[1]))
            rect_height = int(np.linalg.norm(box[0] - box[3]))
            if rect_width <= self.min_width or rect_height <= self.min_height:
                continue
            dt_boxes_new.append(box)
            dt_scores_new.append(score)
        dt_boxes = np.array(dt_boxes_new)
        dt_scores = np.array(dt_scores_new)
        return dt_boxes, dt_scores

    def clip_det_res(self, points, img_height, img_width):
        for pno in range(points.shape[0]):
            points[pno, 0] = int(min(max(points[pno, 0], 0), img_width - 1))
            points[pno, 1] = int(min(max(points[pno, 1], 0), img_height - 1))
        return points

    def execute(self, requests):
        output0_dtype = self.output0_dtype
        responses = []

        for request in requests:
            # Get INPUT0
            shape_lists = pb_utils.get_input_tensor_by_name(
                request, "SHAPE_LISTS"
            ).as_numpy()
            output_infers = pb_utils.get_input_tensor_by_name(
                request, "OUTPUT_INFERS"
            ).as_numpy()
            number_request = shape_lists.shape[0]

            if self.max_box == 1:
                results = (
                    np.zeros((number_request, 4, 2), dtype=np.float32) * -1
                )
            else:
                results = (
                    np.zeros(
                        (number_request, self.max_box, 4, 2), dtype=np.float32
                    )
                    * -1
                )

            image_shape = (output_infers.shape[2], output_infers.shape[3], 3)

            preds = {}
            preds["maps"] = output_infers
            post_result = self.postprocess_op(preds, shape_lists)
            for d in range(number_request):
                dt_boxes = post_result[d]["points"]
                dt_scores = post_result[d]["scores"]
                dt_boxes, dt_scores = self.filter_tag_det_res(
                    dt_boxes, dt_scores, image_shape
                )
                if len(dt_boxes) > 0:
                    dt_scores, dt_boxes = (
                        np.array(list(t))
                        for t in zip(*sorted(zip(dt_scores, dt_boxes)))
                    )
                if self.max_box == 1:
                    if len(dt_boxes) > 0:
                        results[d] = dt_boxes[0]
                else:
                    for j in range(min(self.max_box, dt_boxes.shape[0])):
                        results[d][j] = dt_boxes[j]

            # out_0_np = np.ascontiguousarray(results, dtype=np.float32)
            out_tensor_0 = pb_utils.Tensor(
                "OUTPUT_DETS", results.astype(output0_dtype)
            )
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[out_tensor_0]
            )
            responses.append(inference_response)

        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is OPTIONAL. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print("Cleaning up...")
