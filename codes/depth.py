import cv2
import torch
import urllib.request

import matplotlib.pyplot as plt

model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
#model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
#model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

midas = torch.hub.load("intel-isl/MiDaS", model_type)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform

def calcdepthloss(filename, filename2):
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    input_batch = transform(img).to(device)
    with torch.no_grad():
        prediction = midas(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    output = prediction.cpu().numpy()
    img2 = cv2.imread(filename2)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    input_batch2 = transform(img2).to(device)
    with torch.no_grad():
        prediction2 = midas(input_batch2)

        prediction2 = torch.nn.functional.interpolate(
            prediction2.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    output2 = prediction2.cpu().numpy()
    cv2.imwrite("depth_diff.jpg",img = cv2.cvtColor((output-output2)**2, cv2.COLOR_GRAY2RGB))

#calcdepthloss("/Users/adityasomani/Desktop/3-1/DL/Neural_Image_Transfer/codes/dataset/paris.jpg", "/Users/adityasomani/Desktop/3-1/DL/Neural_Image_Transfer/codes/dataset/starry_night.jpg")


