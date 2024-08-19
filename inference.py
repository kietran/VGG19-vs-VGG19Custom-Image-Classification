import torch
import cv2
import argparse
import numpy as np
from src.model import MyVGG19
import torch.nn as nn
import os


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', '-i', type=str, default=None, required=True)
    parser.add_argument('--image_size', '-s', type=int, default=224)
    parser.add_argument('--checkpoint', '-c', type=str, default='custom_vgg19_checkpoint/best.pt')    
    args = parser.parse_args()
    return args

def inference(args):
    categories = ['Yamaha', 'Honda', 'VinFast', 'Suzuki', 'Others']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MyVGG19(num_classes=len(categories))
    model.to(device)
    if args.checkpoint and os.path.isfile(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, weights_only=True, map_location=device)
        model.load_state_dict(checkpoint)
    else:
        print('No checkpoint found')
        exit(0)

    ori_image = cv2.imread(args.image_path)
    image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (args.image_size, args.image_size))
    image = np.transpose(image, (2, 0, 1))/255.0
    image = torch.from_numpy(image).float()
    image = image[np.newaxis, :]
    image = image.to(device)

    model.eval()
    softmax = nn.Softmax()
    with torch.inference_mode():
        prediction = model(image)
        prediction = softmax(prediction)
        conf_score, class_name = torch.max(prediction, dim=1)
        class_name = categories[class_name]
        cv2.imshow(f"{class_name} with confident is {conf_score.item()}", ori_image)
        cv2.waitKey(0)


if __name__ == "__main__":
    args = get_args()
    inference(args)