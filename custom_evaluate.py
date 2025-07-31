import numpy as np
import pandas as pd
from fddbenchmark import FDDEvaluator
from fddbenchmark.evaluator import cluster_acc
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score


class CustomEvaluator(FDDEvaluator):
    def evaluate_fdd(self, labels, pred):
        # labels should be non-negative integer values, normal is 0
        assert np.all(np.sort(np.unique(labels)) == np.arange(labels.max() + 1))
        # confustion matrix: rows are truth classes, columns are predicted classes
        fdd_cm = confusion_matrix(labels, pred, labels=np.arange(labels.max() + 1))
        metrics = {'detection': dict(), 'diagnosis': dict(), 'clustering': dict(), 'classification': dict()}
        metrics['confusion_matrix'] = fdd_cm
        metrics['detection']['TPR'] = fdd_cm[1:, 1:].sum() / fdd_cm[1:, :].sum()
        metrics['detection']['FPR'] = fdd_cm[0, 1:].sum() / fdd_cm[0, :].sum()

        correct_diagnosis = fdd_cm[1:, 1:].diagonal()
        tp = fdd_cm[1:, 1:].sum()
        metrics['diagnosis']['CDR_total'] = correct_diagnosis.sum() / tp

        correct_classification = fdd_cm.diagonal()
        metrics['classification']['TPR'] = correct_classification / fdd_cm.sum(axis=1)
        metrics['classification']['FPR'] = fdd_cm[0] / fdd_cm[0].sum()

        metrics['clustering']['ACC'] = cluster_acc(labels.values, pred.values)
        metrics['clustering']['NMI'] = normalized_mutual_info_score(labels.values, pred.values)
        metrics['clustering']['ARI'] = adjusted_rand_score(labels.values, pred.values)
        return metrics

    def str_metrics(self, labels, pred):
        metrics = self.evaluate_fdd(labels, pred)
        str_metrics = []
        str_metrics.append('FDD metrics\n-----------------')

        str_metrics.append('TPR/FPR:')
        for i in np.arange(labels.max()).astype('int'):
            str_metrics.append('    Fault {:02d}: {:.4f}/{:.4f}'.format(i + 1, metrics['classification']['TPR'][i + 1],
                                                                        metrics['classification']['FPR'][i + 1]))

        str_metrics.append('Detection TPR: {:.4f}'.format(metrics['detection']['TPR']))
        str_metrics.append('Detection FPR: {:.4f}'.format(metrics['detection']['FPR']))
        # str_metrics.append('Average Detection Delay (ADD): {:.2f}'.format(metrics['detection']['ADD']))
        str_metrics.append('Total Correct Diagnosis Rate (Total CDR): {:.4f}'.format(metrics['diagnosis']['CDR_total']))

        str_metrics.append('\nClustering metrics\n-----------------')
        str_metrics.append('Adjusted Rand Index (ARI): {:.4f}'.format(metrics['clustering']['ARI']))
        str_metrics.append('Normalized Mutual Information (NMI): {:.4f}'.format(metrics['clustering']['NMI']))
        str_metrics.append('Unsupervised Clustering Accuracy (ACC): {:.4f}'.format(metrics['clustering']['ACC']))
        return '\n'.join(str_metrics)

    def evaluate_classification(self, labels, pred):
        return {'accuracy': accuracy_score(labels, pred), 'f1_score': f1_score(labels, pred, average='weighted')}
