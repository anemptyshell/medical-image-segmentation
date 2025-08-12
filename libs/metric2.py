from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import confusion_matrix
import torch

import numpy as np


def Jaccard(pred_arg, mask):
    pred_arg = np.argmax(pred_arg.cpu().data.numpy(), axis=1)
    mask = mask.cpu().data.numpy()

    y_true_f = mask.reshape(mask.shape[0] * mask.shape[1] * mask.shape[2], order='F')
    y_pred_f = pred_arg.reshape(pred_arg.shape[0] * pred_arg.shape[1] * pred_arg.shape[2], order='F')

    intersection = np.float64(np.sum(y_true_f * y_pred_f))
    jac_score = intersection / (np.sum(y_true_f) + np.sum(y_pred_f) - intersection)

    return jac_score


class metric(object):    
    def __init__(self, pred, gt, metric_list):
        super(metric, self).__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.pred = np.array(pred).reshape(-1)   ## 展成一维数组
        self.gt = np.array(gt).reshape(-1)
        self.y_pre = np.where(self.pred >= 0.5, 1, 0)
        self.y_true = np.where(self.gt>=0.5, 1, 0)


        self.confusion_matrix = confusion_matrix(self.y_true, self.y_pre)
        TN, FP, FN, TP = self.confusion_matrix [0,0], self.confusion_matrix [0,1], self.confusion_matrix [1,0], self.confusion_matrix [1,1] 

        self.accuracy = float(TN + TP) / float(np.sum(self.confusion_matrix )) if float(np.sum(self.confusion_matrix )) != 0 else 0
        self.sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
        self.specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
        self.dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
        self.miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0
        self.precision = float(TP) / float(TP + FP) if  float(TP + FP) != 0 else 0
        self.recall = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
        self.f1_score = float(2 * self.precision * self.recall) / float(self.precision + self.recall) if float(self.precision + self.recall) != 0 else 0
        # self.jac = 

        self.metric_list = metric_list

        self.metric_dict = {
            "DSC": self.dsc,
            "ACC": self.accuracy,
            "SEN": self.sensitivity,
            "SPE": self.specificity,
            "IoU": self.miou,
            "PRE": self.precision,
            "recall": self.recall,
            "F1_score": self.f1_score,
            "confusion_matrix": self.confusion_matrix,
        }

        self.metric_values_list = self.get_metric(self.metric_list)

        self.indicator_for_best = None
        self.best = torch.tensor(0.0).to(self.device)
        self.best_trigger = False
        self.higher_is_best = ["DSC", "ACC", "SEN", "SPE", "IoU", "PRE", "recall", "F1_score",]
        self.lower_is_best = [""]


    def metric_for_sorting(self, best, indicator_for_best):
        if indicator_for_best is None:
            for self.indicator_for_best in range(0, len(self.metric_values_list)):   # self.indicator_for_best是索引值0，self.metric_values_list是一个列表
                
                if not self.metric_values_list[self.indicator_for_best] is None:
                    self.best = torch.tensor(0.0).to(self.device) if self.metric_list[self.indicator_for_best] in self.higher_is_best else float(1)
                    break   
        else:
            self.indicator_for_best = indicator_for_best
            self.best = best



    def best_value_indicator(self, best, indicator_for_best):

        self.metric_for_sorting(best, indicator_for_best)

        comparison_operator = "<" if self.metric_list[self.indicator_for_best] in self.higher_is_best else ">"

        exec("self.best_trigger = True if self.best {} self.metric_values_list[self.indicator_for_best] else False".format(comparison_operator))
        self.best = self.metric_values_list[self.indicator_for_best] if self.best_trigger else self.best

        return self.best, self.best_trigger, self.indicator_for_best      # DSC的值，true，0
        


    def get_metric(self, metric_list):

        metric_values_list = []
        for i in range(len(metric_list)):
             
            if self.metric_dict[metric_list[i]] is None:
                raise Exception(self.metric_dict[metric_list[i]] + " can't be compute.")

            if isinstance(self.metric_dict[metric_list[i]], np.ndarray):  # isinstance() 函数来判断一个对象是否是一个已知的类型.指标为数组时候的占位符
                if not self.metric_dict[metric_list[i]].size == 1:
                    metric_values_list.append(None)
                    continue

            metric_values_list.append(self.metric_dict[metric_list[i]])

        return metric_values_list   # 返回存储指标结果的列表





