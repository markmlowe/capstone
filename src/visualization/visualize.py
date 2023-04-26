import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import classification_report


def conf_matr(y_test, preds, score,sup_title):

    cm = metrics.confusion_matrix(y_test, preds)
    plt.figure(figsize=(9,9))
    sns.heatmap(cm, annot=True, fmt="d", linewidths=.5, square = True, cmap = 'Blues_r')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    all_sample_title = 'Accuracy Score: {:.2f}%'.format(score*100, {"3f"})
    plt.title(all_sample_title, size = 15, style='italic')
    plt.suptitle(sup_title, size=18)

    print(classification_report(y_test,preds))

    plt.show()

    return 0