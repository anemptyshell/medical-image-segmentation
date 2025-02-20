from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import confusion_matrix
import torch
from libs.metric_utils.cldice import clDice

import numpy as np


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
        # 在model中，metric_list = self.args.metric，是多个参数被组合成的一个列表，如metric_list=['DSC', 'confusion_matrix']
        # self.metric_values_list是一个包含metric_list中参数的值的列表

        self.indicator_for_best = None
        self.best = torch.tensor(0.0).to(self.device)
        self.best_trigger = False
        self.higher_is_best = ["DSC", "ACC", "SEN", "SPE", "IoU", "PRE", "recall", "F1_score"]
        self.lower_is_best = [""]


    def metric_for_sorting(self, best, indicator_for_best):
        if indicator_for_best is None:
            for self.indicator_for_best in range(0, len(self.metric_values_list)):   # self.indicator_for_best是索引值0，self.metric_values_list是一个列表
                
                if not self.metric_values_list[self.indicator_for_best] is None:
                    self.best = torch.tensor(0.0).to(self.device) if self.metric_list[self.indicator_for_best] in self.higher_is_best else float(1)
                    break   
                    # 终止循环，找到第一个可用的指标后退出
                    # 若某一个指标是越高越好，则将best设为一个浮点数为0.0的张量 否则设为1.0。我的理解是只改变best的值
        else:
            self.indicator_for_best = indicator_for_best
            self.best = best


    def best_value_indicator(self, best, indicator_for_best):
        # 选取所有epoch的指标中的最佳值, indicator_for_best = None

        self.metric_for_sorting(best, indicator_for_best)

        comparison_operator = "<" if self.metric_list[self.indicator_for_best] in self.higher_is_best else ">"

        exec("self.best_trigger = True if self.best {} self.metric_values_list[self.indicator_for_best] else False".format(comparison_operator))
        self.best = self.metric_values_list[self.indicator_for_best] if self.best_trigger else self.best

        return self.best, self.best_trigger, self.indicator_for_best      # DSC的值，true，0
        

    def get_metric(self, metric_list):
        # 获取自己需要的指标；metric_list = self.args.metric，是多个参数被组合成的一个列表，如 metric_list=['DSC', 'confusion_matrix']

        metric_values_list = []
        for i in range(len(metric_list)):
            # 遍历指标列表metric_list的索引
             
            if self.metric_dict[metric_list[i]] is None:
                raise Exception(self.metric_dict[metric_list[i]] + " can't be compute.")
            # 检查在指标字典self.metric_dict中指定的指标metric[i]是否为None，如果是，则抛出异常——无法计算

            if isinstance(self.metric_dict[metric_list[i]], np.ndarray):  # isinstance() 函数来判断一个对象是否是一个已知的类型.指标为数组时候的占位符
                if not self.metric_dict[metric_list[i]].size == 1:
                    metric_values_list.append(None)
                    continue
            # 如果不等于1，说明指标值不是标量，而是具有多个元素的数组，则将指标结果设置为None，并继续下一个指标的处理。

            metric_values_list.append(self.metric_dict[metric_list[i]])

        return metric_values_list   # 返回存储指标结果的列表
    
        # 这段代码的作用是从指标字典self.metric_dict中获取指定指标的值，并将结果以列表形式返回
        # 比如metric = ['DSC', 'confusion_matrix'], 那么metric_values_list = [aaa, bbb]，aaa和bbb分别是对应的值？？？？？？？？？？？
        # 由于有判断语句 if not self.metric_dict[metric_list[i]].cpu().numpy().size == 1， 而混淆矩阵的size不是1，所以metric_values_list只会有ACC的值




