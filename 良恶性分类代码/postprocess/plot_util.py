import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def plot_auc(gts, pred_probs, title=None, save_path=None):
    assert len(gts) == len(pred_probs)
    fpr, tpr, threshold = roc_curve(gts, pred_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(
        fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc
    )
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    if title:
        plt.title(title)
    plt.legend(loc="lower right")
    plt.show()
    if save_path:
        plt.savefig(save_path)

