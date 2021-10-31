from torchvision import models
import torch

# print(dir(models))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

from torchvision import transforms
transform = transforms.Compose([            #[1]
 transforms.Resize(256),                    #[2]
 transforms.CenterCrop(224),                #[3]
 transforms.ToTensor(),                     #[4]
 transforms.Normalize(                      #[5]
 mean=[0.485, 0.456, 0.406],                #[6]
 std=[0.229, 0.224, 0.225]                  #[7]
 )])

 # Import Pillow
from PIL import Image
img = Image.open("E:\\Documents\\Niranjan\\questions\\qure.ai\\inference_test_images\\cat1.jpg")

img_t = transform(img)
batch_t = torch.unsqueeze(img_t, 0)

with open('E:\\Documents\\Niranjan\\questions\\qure.ai\\imagenet1000_clsidx_to_labels.txt') as f:
  labels = [line.strip() for line in f.readlines()]

# First, load the model
resnet = models.resnet18(pretrained=True)

# Second, put the network in eval mode
resnet.eval()

# Third, carry out model inference
out = resnet(batch_t)

# Forth, print the top 5 classes predicted by the model
_, indices = torch.sort(out, descending=True)
percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
print([(labels[idx]) for idx in indices[0][:1]])
