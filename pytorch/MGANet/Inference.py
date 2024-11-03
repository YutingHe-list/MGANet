# -*- coding: utf-8 -*-
import os
from os.path import join
from time import time

import numpy as np
import torch

from models.MGANet import MGANet
import SimpleITK as sitk

class AverageMeter(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def Inference(save_dir='results', test_image_dir='data/test/image', checkpoint_dir='weights', model_name="DenseBiasNet_4c_gws"):
    net_S = MGANet(n_channels=1, n_classes=5, num_learners=4, checkpoint_dir=checkpoint_dir, model_name=model_name)
    net_S.load_state_dict(torch.load('{0}/{1}_epoch_{2}.pth'.format(checkpoint_dir, 'MP_' + model_name+'_aug', str(800))))

    if torch.cuda.is_available():
        net_S = net_S.cuda()

    predict(net_S, save_dir, test_image_dir)

def predict(model, save_path, img_path):
    print("Predict test data")
    model.eval()
    image_filenames = [x for x in os.listdir(img_path) if is_image3d_file(x)]
    for imagename in image_filenames:

        image = sitk.GetArrayFromImage(sitk.ReadImage(join(img_path, imagename)))

        image = np.where(image < 0., 0., image)
        image = np.where(image > 2048., 2048., image)
        image = image.astype(np.float32)
        image = image / 2048.
        image = image.astype(np.float32)
        image = torch.from_numpy(image)
        if torch.cuda.is_available():
            image = image.cuda()
        with torch.no_grad():
            a = time()
            predict = model(image[np.newaxis, np.newaxis, :, :, :]).data.cpu().numpy()

        b = time()
        print(imagename, b-a)

        predict = np.argmax(predict[0], axis=0)
        predict = predict.astype(np.int16)
        sitk.WriteImage(sitk.GetImageFromArray(predict), join(save_path, imagename))

def is_image3d_file(filename):
    return any(filename.endswith(extension) for extension in [".nii.gz"])

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    Inference()