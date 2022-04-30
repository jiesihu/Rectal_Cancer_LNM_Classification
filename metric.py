###############################################################################################################
# model diagnosis
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
import pandas as pd
import math
from sklearn import metrics
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
def calculateModelScores(y_true, y_pred,y_prob, dataset='Train',verbal = True):
    """
    calculate the ['accuracy', 'precision', 'recall (senstitive)', 'recall_neg (specificity)', 'f1_score', 'auc', 'threshold', 'brier_score']
    Args:
        y_true: (numpy array (n,)) the label
        y_pred: (numpy array (n,)) the prediction
        y_prob: (numpy array (n,)) the predict probability
        dataset: (str) Name for figure
        verbal: (bool) Whether to show the figure
    """
    
    # classification metrics
    scores = {}

    # print report
    target_names = ['class 0', 'class 1']
    text = metrics.classification_report(y_true, y_pred, target_names=target_names)
    
    #pdb.set_trace()
    conf_mat=pd.crosstab(y_true, y_pred,rownames=['label'],colnames=['pre'])
    if verbal:
        print(conf_mat)
        print(text)

    # accuracy
    scores['accuracy'] = metrics.accuracy_score(y_true, y_pred)

    # precision
    scores['precision'] = metrics.precision_score(y_true, y_pred)

    # recall
    scores['recall (senstitive)'] = metrics.recall_score(y_true, y_pred)
    
    scores['recall_neg (specificity)'] = metrics.recall_score(y_true==0, y_pred==0)

    # F1-score
    scores['f1_score'] = metrics.f1_score(y_true, y_pred)

    # ROC/AUC
    fpr, tpr, th = metrics.roc_curve(y_true, y_prob)
    scores['auc'] = metrics.auc(fpr, tpr)
    
    threshold=th[np.argmax(tpr - fpr)]
    scores['threshold']=threshold
    if verbal:
        print('figure ROC curve')
        figureROC(fpr, tpr, th, scores['auc'], dataset)

        # RadScore
        figureRadScore(y_true, y_prob, 'lr', dataset,threshold)


    # brier_score_loss
    scores['brier_score'] = metrics.brier_score_loss(y_true, y_prob, pos_label=y_true.max())
    return scores

def figureROC(fpr, tpr, th, auc, dataset):
    # calculate the best cut-off point by Youden index
    uindex = np.argmax(tpr - fpr)

    plt.figure()
    plt.plot(fpr, tpr, lw=1, label='ROC (area = %0.2f)' % auc)
    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
    plt.plot(fpr[uindex], tpr[uindex], 'r', markersize=8)
    plt.text(fpr[uindex], tpr[uindex], '%.3f(%.3f,%.3f)' % (th[uindex], fpr[uindex], tpr[uindex]), ha='center', va='bottom', fontsize=10)
    plt.title('ROC curve (' + dataset + ')')
    plt.legend(loc='lower right')
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.xlabel('1-Specificity')
    plt.ylabel('Sensitivity')
    plt.grid(True)
    plt.show()
def figureRadScore(y_true, y_prob, clf_name, dataset,threshold):
    if clf_name == 'lr':
        rad_score = np.log(y_prob / (1-y_prob))
        adjust=np.log(threshold/(1-threshold))
        rad_score_sort = np.sort(rad_score-adjust)
        y_true_sort = y_true[np.argsort(rad_score)]

        # figure rad score
        print("figure RadScore")
        plt.figure()
        class0 = np.where(y_true_sort == 0)[0]
        class1 = np.where(y_true_sort == 1)[0]
        
        plt.bar(np.array(class0).squeeze(), rad_score_sort[class0], width=0.6, color='magenta', label='0')
        plt.bar(np.array(class1).squeeze(), rad_score_sort[class1], width=0.6, color='cyan', label='1')
        plt.title('Rad-Score (' + dataset + ')')
        plt.xlabel('Bar Chart')
        plt.ylabel('Rad Score')
        #plt.grid(True)
        plt.xticks(range(len(np.argsort(rad_score))), np.argsort(rad_score))
        plt.legend(loc='lower right')
        plt.show()
