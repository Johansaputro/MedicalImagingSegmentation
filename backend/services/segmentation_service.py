import os

import torch
import torch.nn.functional as F

import numpy as np
import SimpleITK as sitk
import scipy.ndimage as ndimage

import logging

from config import NETWORK, RESULT_DIR
from time import time

class SegmentationService:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

        self.upper = 350
        self.lower = -self.upper
        self.down_scale = 0.5
        self.size = 48
        self.slice_thickness = 3

    def predict_nifti(self, filepath):
        self.logger.info("predict_nifti. Start predicting nifti file in {}".format(filepath))
        ct = sitk.ReadImage(filepath, sitk.sitkInt16)
        ct_array = sitk.GetArrayFromImage(ct)

        original_shape = ct_array.shape

        ct_array = ndimage.zoom(ct_array, (ct.GetSpacing()[-1] / self.slice_thickness, self.down_scale, self.down_scale), order=3)

        # slice and sample in the axial direction
        flag  =  False
        start_slice = 0
        end_slice = start_slice + self.size - 1
        ct_array_list = []

        while end_slice <= ct_array.shape[0] - 1:
            ct_array_list.append(ct_array[start_slice:end_slice + 1, :, :])

            start_slice = end_slice + 1
            end_slice = start_slice + self.size - 1

        # When it is not divisible, take the last block in reverse
        if end_slice is not ct_array.shape[0] - 1:
            flag = True
            count = ct_array.shape[0] - start_slice
            ct_array_list.append(ct_array[-self.size:, :, :])

        outputs_list = []
        with torch.no_grad():
            for ct_array in ct_array_list:

                ct_tensor = torch.FloatTensor(ct_array)
                ct_tensor = ct_tensor.unsqueeze(dim=0)
                ct_tensor = ct_tensor.unsqueeze(dim=0)

                outputs = NETWORK(ct_tensor)
                outputs = outputs.squeeze()

                outputs_list.append(outputs.cpu().detach().numpy())
                del outputs

        # Start splicing results after execution
        pred_seg = np.concatenate(outputs_list[0:-1], axis=1)
        if flag is False:
            pred_seg = np.concatenate([pred_seg, outputs_list[-1]], axis=1)
        else:
            pred_seg = np.concatenate([pred_seg, outputs_list[-1][:, -count:, :, :]], axis=1)

        pred_seg = torch.FloatTensor(pred_seg).unsqueeze(dim=0)
        pred_seg = F.interpolate(pred_seg, original_shape, mode='trilinear').squeeze().detach().numpy()
        pred_seg = np.argmax(pred_seg, axis=0)
        pred_seg = np.round(pred_seg).astype(np.uint8)

        pred_seg = sitk.GetImageFromArray(pred_seg)

        pred_seg.SetDirection(ct.GetDirection())
        pred_seg.SetOrigin(ct.GetOrigin())
        pred_seg.SetSpacing(ct.GetSpacing())

        file_destination = os.path.abspath(os.path.join(RESULT_DIR, "result.nii.gz"))

        sitk.WriteImage(pred_seg, file_destination)
        self.logger.info("predict_nifti. Predicted File Saved in {}".format(file_destination))
        del pred_seg

        return file_destination
    
