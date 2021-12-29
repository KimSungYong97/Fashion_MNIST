from itertools import chain
import sklearn.metrics as metrics
def show(predictions_list,labels_list,confusion_matrix):

    predictions_l = [predictions_list[i].tolist() for i in range(len(predictions_list))]
    labels_l = [labels_list[i].tolist() for i in range(len(labels_list))]
    predictions_l = list(chain.from_iterable(predictions_l))
    labels_l = list(chain.from_iterable(labels_l))

    confusion_matrix(labels_l, predictions_l)
    print("Classification report for CNN :\n%s\n"
          % (metrics.classification_report(labels_l, predictions_l)))