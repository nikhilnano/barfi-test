import os
import argparse
from numpy.core.defchararray import count
import requests
import torch
import torch.nn as nn
from torchvision import transforms
import videotransforms
import numpy as np
import torch.nn.functional as F
from pytorch_i3d import InceptionI3d
from datasets.nslt_dataset_all import NSLT as Dataset, get_num_class
from cv2 import cv2

class Predictor:
    def __init__(self):
        myfile = open(r'preprocess/wlasl_class_list.txt', 'r')
        self.image_path = 'new_pred.MP4'
        i = 0
        self.dict_of_labels = {}
        for line in myfile.readlines():
            i+=1
            values = line.split('\t')
            self.dict_of_labels[values[0]] = values[1][:-1]

        num_class = 2000
        self.i3d = InceptionI3d(400, in_channels=3)
        self.i3d.load_state_dict(torch.load('weights/rgb_imagenet.pt', map_location=torch.device('cpu')))
        self.i3d.replace_logits(num_class)
        self.i3d.load_state_dict(torch.load(r"archived/asl2000/FINAL_nslt_2000_iters=5104_top1=32.48_top5=57.31_top10=66.31.pt", map_location=torch.device('cpu')))
        self.i3d.eval()

    def download_from_firebase(self, url):
        i=0
        r = requests.get(url, stream=True)
        if r.status_code==200:
            with open(self.image_path, 'wb') as f:
                for chunk in r:
                    f.write(chunk)
        return self.image_path

    def load_rgb_frames_from_video(self, video_path, start=0, num=-1):
        vidcap = cv2.VideoCapture(video_path)

        frames = []

        vidcap.set(cv2.CAP_PROP_POS_FRAMES, start)
        if num == -1:
            num = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

        for offset in range(num):
            success, img = vidcap.read()

            w, h, c = img.shape
            sc = 224 / w
            img = cv2.resize(img, dsize=(0, 0), fx=sc, fy=sc)

            img = (img / 255.) * 2 - 1

            frames.append(img)
            
        ans = np.asarray(frames, dtype=np.float32)
        ans=torch.Tensor(ans)
        input_list = [ans]
        transforms_test = transforms.Compose([videotransforms.CenterCrop(224)])
        transpose_t = transforms_test(ans)
        transpose_t_input = transpose_t.permute(3,0,1,2)
        input_for_torch_model = transpose_t_input[None, :, :] 
        return input_for_torch_model

    def get_best_class_for_video(self, input_tensor):
        logits = self.i3d(input_tensor)
        predictions = torch.max(logits, dim=2)[0]
        out_labels = np.argsort(predictions.cpu().detach().numpy()[0])
        out_probs = np.sort(predictions.cpu().detach().numpy()[0])
        top_5_labels = out_labels[-5:]
        top_5_probs = out_probs[-5:]
        best_match = torch.argmax(predictions[0]).item()
        best_label = self.dict_of_labels[str(best_match)]

        return best_label, [(self.dict_of_labels[str(top_5_labels[-i])], top_5_probs[-i]) for i in range(1,6)]

    def get_final_labels_in_video(self, url):
        dict_top_5_levels = {}
        final_labels =[]
        
        video_path = self.download_from_firebase(url)
        input_tensors_list = self.load_rgb_frames_from_video(video_path, start=0, num=-1)
        prev_label, prev_5_prev_label = self.get_best_class_for_video(input_tensors_list)
        dict_top_5_levels[prev_label] = prev_5_prev_label
        
        final_labels.append(prev_label)
        os.remove(self.image_path)
        ans_dict = {}
        for (k, v) in dict_top_5_levels.items():
            for i in v:
                ans_dict[i[0]] = f'{i[1]:.2f}'
        return ans_dict

if __name__ == '__main__':
    pred = Predictor()
    top_5_pred = pred.get_final_labels_in_video(
        "https://firebasestorage.googleapis.com/v0/b/barfi-5faf3.appspot.com/o/a%20lot.mp4?alt=media&token=44161bcb-49be-463f-a98a-eb9a99b842e9")
    print(top_5_pred)
