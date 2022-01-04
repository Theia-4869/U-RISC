import os
import numpy as np
import cv2

def F_score(predicted, target, beta=1.0):
    true_positive = np.sum(predicted * target)
    if true_positive == 0:
        return 0, 0, 0
    precision = true_positive / np.sum(predicted).item()
    recall = true_positive / np.sum(target).item()
    score = (1 + beta * beta) * precision * recall / (beta * beta * precision + recall)
    return score, precision, recall


if __name__ == "__main__":
    srcdir = os.path.join("outdir", "simple", "val")
    tardir = os.path.join("data/datasets", "simple", "label", "val")
    outdir = os.path.join("outdir", "simple", "val_mor")

    total_f_score_, total_precison_, total_recall_ = 0.0, 0.0, 0.0
    total_f_score, total_precison, total_recall = 0.0, 0.0, 0.0
    for filename in os.listdir(srcdir):
        mask = 1.0 - cv2.imread(os.path.join(srcdir, filename), cv2.IMREAD_GRAYSCALE) / 255
        label = 1.0 - cv2.imread(os.path.join(tardir, filename), cv2.IMREAD_GRAYSCALE) / 255
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1,1))
        open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        # mask_close = cv2.dilate(mask, kernel)
        # mask_close = cv2.erode(mask, kernel)
        mask_close = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=1)
        # mask_close = cv2.morphologyEx(mask_close, cv2.MORPH_OPEN, open_kernel, iterations=1)

        f_score_, precision_, recall_ = F_score(mask, label)
        f_score, precision, recall = F_score(mask_close, label)
        total_f_score_ += f_score_
        total_precison_ += precision_
        total_recall_ += recall_
        total_f_score += f_score
        total_precison += precision
        total_recall += recall
        print(f_score_)
        print(f_score)

        mask_close = (1.0 - mask_close)*255
        cv2.imwrite(os.path.join(outdir, filename), mask_close)

    mean_f_score_ = total_f_score_ / len(os.listdir(srcdir))
    mean_f_score = total_f_score / len(os.listdir(srcdir))
    print(mean_f_score_)
    print(mean_f_score)
