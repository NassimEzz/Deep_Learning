# YoloV1 in Pytorch

This repository is a guided implementation of YoloV1 from scratch. It is
inspired by (and borrows material from) [Aladdin Persson Youtube video](https://youtube.com/watch?v=n9_XyCGr-MI).
The implementation steps are exposed as a set of exercises, presented and exposed in the [jupyter notebook](./Yolo.ipynb).

# Guidelines to run within a Google Colab runtime
In order to complete the lab within a Colab runtime, you will need to add the following code at the very begining of the notebook:
```python
import os
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
if not os.path.exists('/content/drive/MyDrive/YoloV1Notebook'):
  print('====== Cloning the github repository')
  os.chdir('/content/drive/MyDrive')
  !git clone https://github.com/DrLSimon/YoloV1Notebook
os.chdir('/content/drive/MyDrive/YoloV1Notebook/')
```

Note that you will need a google account in order to have access to Google Drive.
