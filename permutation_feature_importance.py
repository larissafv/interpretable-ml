import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_curve

def get_optimal_precision_recall(y_true, y_score):
    """Find precision and recall values that maximize f1 score."""
    n = np.shape(y_true)[1]
    opt_precision = []
    opt_recall = []
    opt_f1_score = []
    opt_threshold = []
    for k in range(n):
        # Get precision-recall curve
        precision, recall, threshold = precision_recall_curve(y_true[:, k], y_score[:, k])
        # Compute f1 score for each point (use nan_to_num to avoid nans messing up the results)
        f1_score = np.nan_to_num(2 * precision * recall / (precision + recall))
        # Select threshold that maximize f1 score
        index = np.argmax(f1_score)
        opt_f1_score.append(f1_score[index])
        opt_precision.append(precision[index])
        opt_recall.append(recall[index])
        t = threshold[index-1] if index != 0 else threshold[0]-1e-10
        opt_threshold.append(t)
    return np.array(opt_precision), np.array(opt_recall), np.array(opt_f1_score), np.array(opt_threshold)

def change_input(ecgs, lead, idx, mod):
  final_input = []
  for ecg in ecgs:
    aux = ecg.copy()
    for i in range(64):
      aux[idx+i][lead] = mod
    final_input.append(aux)
  return np.array(final_input)

def normalize_score(y_true, y_score):
  _, _, _, threshold = get_optimal_precision_recall(y_true, y_score)
  mask = y_score > threshold
  y_pred = np.zeros_like(y_score)
  y_pred[mask] = 1
  return y_pred

def permutation_feature_importance(ecgs, model, y_pred, feat_max=10.0, feat_min=-10.0):
  all_acc = []
  for intervalo in range(768):
    n = intervalo * 64
    lead = int(n/4096)
    idx = n - (4096*lead)
    acc = []
    for mod in [feat_max, feat_min]:
      new_input = change_input(ecgs, lead, idx, mod)
      y_new = model.predict(new_input, batch_size=10, verbose=1)
      y_new = normalize_score(y_pred, y_new)
      acc.append(accuracy_score(y_pred, y_new))
    all_acc.append(np.mean(np.array(acc)))
  return all_acc