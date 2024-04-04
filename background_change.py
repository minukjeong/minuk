import cv2
import matplotlib.pyplot as plt
from segment_anything import SamPredictor, sam_model_registry
from segment_anything import SamAutomaticMaskGenerator
#from google.colab.patches import cv2_imshow
import torch
import numpy as np

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

IMAGE_PATH="/home/snu/Desktop/sam/images/object/asdf.jpg"
BACKGROUND_PATH="/home/snu/Desktop/sam/images/background/snow.jpg"

img=cv2.imread(IMAGE_PATH)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
background = cv2.imread(BACKGROUND_PATH)
background=cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
background = cv2.resize(background, (img.shape[1], img.shape[0]))
sam = sam_model_registry["vit_h"](checkpoint="/home/snu/Desktop/sam/sam_vit_h_4b8939.pth")
sam.to(device=DEVICE)

mask_generator = SamAutomaticMaskGenerator(sam)
result = mask_generator.generate(img_rgb)

type(result[0]['segmentation'][0][0])

np.unique(result[0]['segmentation'])

for i in range(0,len(result)):
  result[i]['segmentation']=result[i]['segmentation'].astype(np.uint8)

for i in range(0,len(result)):
  img_test=result[i]['segmentation']
  print("Area - ",result[i]['area'])
#  plt.imshow(img_test)
#  plt.show()

mask=result[4]['segmentation']

mask_boolean = mask > 0

foreground = np.zeros_like(img_rgb)
for c in range(0, 3):  # RGB 채널에 대해 반복
    foreground[:, :, c] = img_rgb[:, :, c] * mask_boolean

background[mask_boolean] = foreground[mask_boolean]

plt.imshow(background)

res=cv2.cvtColor(background, cv2.COLOR_RGB2BGR)

cv2.imwrite("output_image.png",res)
