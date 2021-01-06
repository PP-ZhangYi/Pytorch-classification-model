import os
import torch
from torchvision import transforms
from PIL import Image
import time

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
img_size = 224
model = torch.load('./model/epoch0_acc0.9044_loss0.2180.pt')

data_transforms = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def predict_image(img_path):
    start_time = time.time()
    img = Image.open(img_path)
    input = data_transforms(img).unsqueeze(0)
    input.to(device)
    output = model(input)
    _, predict = torch.max(output, 1)
    end_time = time.time()
    print('use_time', end_time - start_time)
    print('precicted classes: ', predict.numpy()[0])


while True:
    img_path = input('please input img_path:')
    if not os.path.exists(img_path) and img_path != 'q':
        print("The path error, Try again!")
        continue
    if img_path == 'q':
        break
    predict_image(img_path)
