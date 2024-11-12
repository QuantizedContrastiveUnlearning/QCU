'''
MIA Train (label: data)
  - Member: Df, Dr
  - Non-Member: Dt
MIA Eval (label: data)
  - Member: Df
  - Non-Member: Dt
'''
import torch
import torch.nn as nn

import random
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit

def cm_score(estimator, X, y):
    y_pred = estimator.predict(X)
    cnf_matrix = confusion_matrix(y, y_pred)
    
    FP = cnf_matrix[0][1] 
    FN = cnf_matrix[1][0] 
    TP = cnf_matrix[0][0] 
    TN = cnf_matrix[1][1]


    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    # Specificity or true negative rate
    TNR = TN/(TN+FP) 
    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    # Negative predictive value
    NPV = TN/(TN+FN)
    # Fall out or false positive rate
    FPR = FP/(FP+TN)
    # False negative rate
    FNR = FN/(TP+FN)
    # False discovery rate
    FDR = FP/(TP+FP)

    # Overall accuracy
    ACC = (TP+TN)/(TP+FP+FN+TN)
    print (f"FPR:{FPR:.2f}, FNR:{FNR:.2f}, FP{FP:.2f}, TN{TN:.2f}, TP{TP:.2f}, FN{FN:.2f}")
    return ACC


def evaluate_attack_model(sample_loss,
                          members,
                          n_splits = 5,
                          random_state = None):
  """Computes the cross-validation score of a membership inference attack.
  Args:
    sample_loss : array_like of shape (n,).
      objective function evaluated on n samples.
    members : array_like of shape (n,),
      whether a sample was used for training.
    n_splits: int
      number of splits to use in the cross-validation.
    random_state: int, RandomState instance or None, default=None
      random state to use in cross-validation splitting.
  Returns:
    score : array_like of size (n_splits,)
  """

  unique_members = np.unique(members)
  if not np.all(unique_members == np.array([0, 1])):
    raise ValueError("members should only have 0 and 1s")

  attack_model = LogisticRegression(random_state=random_state)
  # attack_model = SVC(C=3, gamma="auto", kernel="rbf", random_state=random_state)
  cv = StratifiedShuffleSplit(
      n_splits=n_splits, random_state=random_state)
  return cross_val_score(attack_model, sample_loss, members, cv=cv, scoring=cm_score)

def collect_losses(model, cr, dataloader):
    device = next(model.parameters()).device
    losses = []
    dataloader = torch.utils.data.DataLoader(dataloader.dataset, num_workers=dataloader.num_workers, batch_size=128, shuffle=False)
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = cr(output, target)
        losses = losses + list(loss.cpu().detach().numpy())
    return losses

def membership_inference_attack(model, data_loader, seed):
    t_loader, f_loader = data_loader['test'], data_loader['forget']
    try:
      fgt_cls = list(np.unique(f_loader.dataset.targets))
      indices = [i in fgt_cls for i in t_loader.dataset.targets]
      t_loader.dataset.data = t_loader.dataset.data[indices]
      t_loader.dataset.targets = t_loader.dataset.targets[indices]

    except:
      fgt_cls = list(np.unique(f_loader.dataset.labels))
      indices = [i in fgt_cls for i in t_loader.dataset.labels]
      t_loader.dataset.data = t_loader.dataset.data[indices]
      t_loader.dataset.labels = t_loader.dataset.labels[indices]

    cr = nn.CrossEntropyLoss(reduction='none')
    model.eval()

    test_losses = collect_losses(model, cr, t_loader)
    forget_losses = collect_losses(model, cr, f_loader)

    np.random.seed(seed)
    random.seed(seed)
    if len(forget_losses) > len(test_losses):
        forget_losses = list(random.sample(forget_losses, len(test_losses)))
    elif len(test_losses) > len(forget_losses):
        test_losses = list(random.sample(test_losses, len(forget_losses)))

    print(np.max(test_losses), np.min(test_losses))
    print(np.max(forget_losses), np.min(forget_losses))

    test_labels = [0]*len(test_losses)
    forget_labels = [1]*len(forget_losses)
    features = np.array(test_losses + forget_losses).reshape(-1,1)
    labels = np.array(test_labels + forget_labels).reshape(-1)
    features = np.clip(features, -100, 100)
    score = evaluate_attack_model(features, labels, n_splits=5, random_state=seed)

    return score.mean() * 100