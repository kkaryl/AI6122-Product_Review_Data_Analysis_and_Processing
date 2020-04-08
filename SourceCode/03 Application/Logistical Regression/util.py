from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score


def get_f1_score(y_true, y_pred):
    return f1_score(y_true, y_pred, labels=[0,1,2],average='macro')

def get_precision_score(y_true, y_pred):
    return precision_score(y_true, y_pred, labels=[0,1,2], average='macro')

def get_recall_score(y_true, y_pred):
    return recall_score(y_true, y_pred, labels=[0,1,2], average='macro')

def get_acc_score(y_true, y_pred):
    return accuracy_score(y_true, y_pred)


def print_score(pred_label,ground_truth):
    f1_score = get_f1_score(ground_truth,pred_label)
    print("f1-score")
    print(f1_score)
    precision = get_precision_score(ground_truth,pred_label)
    print("precision")
    print(precision)
    recall = get_recall_score(ground_truth,pred_label)
    print("recall")
    print(recall)
    acc = get_acc_score(ground_truth,pred_label)
    print("acc")
    print(acc)