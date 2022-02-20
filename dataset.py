"""
Creates a Pytorch dataset to load the Pascal VOC dataset
"""

import torch
import os
import pandas as pd
from PIL import Image


class VOCDataset(torch.utils.data.Dataset):
    def __init__(
        self, csv_file, img_dir, label_dir, S=7, C=20, transform=None, nb_samples=None
    ):
        self.annotations = pd.read_csv(csv_file)
        if nb_samples is not None and nb_samples < len(self.annotations):
            self.annotations = self.annotations[:nb_samples]
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.S = S
        self.C = C

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        boxes = []
        with open(label_path) as f:
            for label in f.readlines():
                class_label, x, y, width, height = [
                    float(x) if float(x) != int(float(x)) else int(x)
                    for x in label.replace("\n", "").split()
                ]

                boxes.append([class_label, x, y, width, height])

        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = Image.open(img_path)
        boxes = torch.tensor(boxes)

        if self.transform:
            # image = self.transform(image)
            image, boxes = self.transform(image, boxes)

        label_matrix = self.__convert_boxes_to_matrix(boxes)
        return image, label_matrix

    def __convert_boxes_to_matrix(self, boxes):
        '''
        Exercise 2: Complete the implementation of this method.
        This function converts a list of `n` boxes into a torch 3d tensor of
        size `S x S x (5+C)`.
        Each box in the list is stored in the tensor in the cell which contains
        the center of box, if this cell is not already taken : in practice, this
        means that some supervision is lost when 2 or more boxes are on the same cell. 
        Each cell is encoded in the following way:
           * the C first components encode the class of the box (1-hot vector)
           * the next value is the confidence and is always 1 here
           * the following for values are the box spatial extension (x,y, w, h)
        Beware : the spatial extension should be expressed relatively to the cell extension,
        while the input values (i.e. in `boxes`) are expressed relatively to the image extension.
        Some conversion is therefore needed.
        '''
        label_matrix = torch.zeros((self.S, self.S, self.C + 5))
        for box in boxes:
            class_label, x, y, width, height = box.tolist()
            class_label = int(class_label)
            ''' todo replace the next 3 instructions to:
               * compute which cell (i,j) contains the center of the box
               * express the box relatively to that cell instead of the entire image
               * store that information  and the class in `label_matrix[i,j,:]`
            '''
            i, j = int(self.S * y), int(self.S * x)
            x_cell, y_cell = self.S * x - j, self.S * y - i
            
            width_cell, height_cell = (
                width * self.S,
                height * self.S,
            )
            
            if label_matrix[i, j, 20] == 0:
                # Set that there exists an object
                label_matrix[i, j, 20] = 1

                # Box coordinates
                box_coordinates = torch.tensor(
                    [x_cell, y_cell, width_cell, height_cell]
                )

                label_matrix[i, j, 21:25] = box_coordinates

                # Set one hot encoding for class_label
                label_matrix[i, j, class_label] = 1
                
            #label_matrix[i,j,:] = torch.arange(self.C+5)


        return label_matrix
