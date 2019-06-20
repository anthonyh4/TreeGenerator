import cv2
from PIL import Image
import numpy as np
import os
import random

ORIG_DIR = './renders'
PREDICTION_DIR = './predictions'

center_crop_amount = 430

def main():
    n = 8
    pairs = [(fn, os.path.join(ORIG_DIR, fn), os.path.join(ORIG_DIR,fn)) for fn in set(os.listdir(ORIG_DIR)).intersection(set(os.listdir(PREDICTION_DIR))) if fn.endswith('.png')]
    random.shuffle(pairs)
    for (i, (fn, orig, pred)) in enumerate(pairs[:n]):
        orig = cv2.imread(orig)
        pred = cv2.imread(pred)
        w, h, c = orig.shape
        startw = w//2-center_crop_amount//2
        endw = startw + center_crop_amount
        starth = h // 2 - center_crop_amount // 2
        endh = starth + center_crop_amount
        orig = orig[startw:endw,starth:endh,:]
        pred = pred[startw:endw,starth:endh,:]
        img = np.concatenate([orig, pred], axis=1)
        #cv2.imshow(fn,img)
        print("predicting")
        cv2.imwrite('results/left_orig_vs_right_prediction%d.png' % i, img)
    #cv2.waitKey(0)

if __name__ == "__main__":
    main()
