"""
Implementation of Yolo Loss Function from the original yolo paper

"""

import torch
import torch.nn as nn
from utils import intersection_over_union


class YoloLoss(nn.Module):
    """
    Calculate the loss for yolo (v1) model
    """

    def __init__(self, S=7, B=2, C=20):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")

        """
        S is split size of image (in paper 7),
        B is number of boxes (in paper 2),
        C is number of classes (in paper and VOC dataset is 20),
        """
        self.S = S
        self.B = B
        self.C = C

        # These are from Yolo paper, signifying how much we should
        # pay loss for no object (noobj) and the box coordinates (coord)
        self.lambda_bck = 0.5
        self.lambda_reg = 5

    def forward(self, predictions, target):
        '''
        Exercise 4: Complete the loss computation.
        '''
        # predictions are shaped (BATCH_SIZE, S*S(C+B*5) when inputted
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B * 5)



        class_loss  =0
        reg_loss = 0
        obj_loss = 0
        bck_loss = 0


        loss = (
            self.lambda_reg * reg_loss
            + obj_loss  
            + self.lambda_bck * bck_loss
            + class_loss  
        )

        return loss
