import torch
print("Cuda available {} Number of Devices {}".format(torch.cuda.is_available(), torch.cuda.device_count()))

import cv2
