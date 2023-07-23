# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from operator import index
import numpy as np
import sys
import json
import io
import cv2
import copy
# triton_python_backend_utils is available in every Triton Python model. You
# need to use this module to create inference requests and responses. It also
# contains some utility functions for extracting information from model_config
# and converting Triton input/output types to numpy types.
import triton_python_backend_utils as pb_utils
from tool import *
import os


class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        self.model_config= json.loads(args['model_config'])

        # Get OUTPUT0 configuration
        output0_config = pb_utils.get_output_config_by_name(
            self.model_config, "INPUT_RECS")
        self.output0_dtype = pb_utils.triton_string_to_numpy(
            output0_config['data_type'])
        
        output1_config = pb_utils.get_output_config_by_name(
            self.model_config, "INDEXS")
        self.output1_dtype = pb_utils.triton_string_to_numpy(
            output1_config['data_type'])
        self.drop_score=0.3     
        self.batch_num = 8
        self.image_shape_rec = (100,48,3)
    

    def execute(self, requests):
        responses = []

        for request in requests:
            # Get INPUT0
            imgs = pb_utils.get_input_tensor_by_name(
                request, "IMAGES").as_numpy()
            imgs=np.transpose(imgs, (0,2, 3,1))
            dt_boxes = pb_utils.get_input_tensor_by_name(
                request, "OUTPUT_DETS").as_numpy()
            
            img_list = []
            indexs=[[0]for i in range(len(imgs))]
            
            for i in range(len(dt_boxes)):
                if(sum(sum(dt_boxes[i]))==0):
                    image_black = np.zeros(self.image_shape_rec, dtype=np.uint8)
                    img_list.append(image_black)
                    indexs[i]=[0]
                    continue
                tmp_box = copy.deepcopy(dt_boxes[i])
                img_crop = get_rotate_crop_image(imgs[i], tmp_box)
                img_list.append(img_crop)
                indexs[i]=[1]                
                # cv2.imwrite("image_crop{}.jpg".format(i), img_crop)
         
            
            
            img_num = len(img_list)
            width_list = []
            for img in img_list:
                width_list.append(img.shape[1] / float(img.shape[0]))
            indices = np.argsort(np.array(width_list))
            st = time.time()
            for beg_img_no in range(0, img_num, self.batch_num):
                end_img_no = min(img_num, beg_img_no + self.batch_num)
                norm_img_batch = []
                max_wh_ratio = 0
                for ino in range(beg_img_no, end_img_no):
                    h, w = img_list[indices[ino]].shape[0:2]
                    wh_ratio = w * 1.0 / h
                    max_wh_ratio = max(max_wh_ratio, wh_ratio)
                for ino in range(beg_img_no, end_img_no):
                        norm_img = resize_norm_img(img_list[indices[ino]],
                                                        max_wh_ratio)
                        norm_img = norm_img[np.newaxis, :]
                        norm_img_batch.append(norm_img)
                    
                norm_img_batch = np.concatenate(norm_img_batch)
               
            
            
            out_0_np = np.ascontiguousarray(norm_img_batch, dtype=np.float32)
            out_tensor_0 = pb_utils.Tensor("INPUT_RECS",
                                           out_0_np.astype(self.output0_dtype))
            indexs=np.array(indexs)
            out_1_np = np.ascontiguousarray(indexs, dtype=np.int32)
            out_tensor_1 = pb_utils.Tensor("INDEXS",
                                           out_1_np.astype(self.output1_dtype))
           
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[out_tensor_0,out_tensor_1])
            responses.append(inference_response)

        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is OPTIONAL. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print('Cleaning up...')
