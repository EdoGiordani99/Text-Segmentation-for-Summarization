import nltk
import torch
import numpy as np
import matplotlib.pyplot as plt

from collections import Counter
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report


def clean_prediction(l):
    """
    This function transforms a list in the following way:
    in:  [1, 1, 0, 1, 1, 1, 2, 2, 1, 2, 2, 2, 3, 3, 0, 0, 1, 0, 3, 0]
    out: [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 0, 0, 0, 0, 0, 0]
    Is like we are applying a low pass filter on the list, to smooth the results
    """
    new = []
    actual = 1

    for i in range(len(l)):

        if i < 4:
            new.append(1)
            continue

        if l[i] == actual:
            new.append(l[i])

        # Check in the span of 9 predictions
        elif l[i] != actual:
            c = Counter(l[i - 4:i + 5])
            try:
                mc = c.most_common()[0][0]  # most common
            except:
                print('FIRST: ', i, l, c)
                raise ValueError(c.most_common())

            if actual != 0 and actual != 3 and mc == 0:
                try:
                    mc = c.most_common()[1][0]  # second most common
                except:
                    print('SECOND: ', i, l, c)
                    raise ValueError(c.most_common())

            if mc == actual:
                new.append(actual)
            else:
                actual = mc
                new.append(mc)

    return new


# ----------------------------------------------
#            EVALUATION FUNCTIONS
# ----------------------------------------------

def transform(a):
    """
    Transforms a segmentation labels list in the following format:
    [1,1,1,2,2,2,3,3,3] --> [0,0,0,1,0,0,1,0,0]
    This is needed to compute the pk score
    """
    out = ''
    prev = a[0]
    for num in a:

        if num != prev:
            prev = num
            out += str(1)
        else:
            out += str(0)

    return out


def compute_pk(labels, preds):
    """
    Given batch labels and predictions, it returns fk score.
    Args:
      labels (torch.Tensor): y batch
      preds (torch.Tensor): prediction for the batch
      average (str): way of computing score over classes
    """

    pk_score = 0
    num = 0

    try:
        labels = labels.tolist()
        preds = preds.tolist()
    except:
        pass

    for lab, pred in zip(labels, preds):

        num += 1
        pk_score += nltk.pk(transform(lab), transform(pred), 1)


    return round(pk_score/num, 4)


def compute_f1(labels, preds, average='micro'):
    """
    Given batch labels and predictions, it returns f1 score.
    Args:
      labels (torch.Tensor): y batch
      preds (torch.Tensor): prediction for the batch
      average (str): way of computing score over classes
    """

    all_preds = []
    all_labels = []

    for pred, label in zip(preds.cpu(), labels.cpu()):
        all_preds += pred
        all_labels += label

    f1 = f1_score(all_labels, all_preds, average=average)

    return round(f1, 4)


# ----------------------------------------------
#         PLOTS AND RESULTS FUNCTIONS
# ----------------------------------------------

def evaluate_model(test_dataset, model, use_correction, scale=7, confusion=True, report=True):
    """
    This function evaluates the model on the test dataset, returning a classification report
    and the relative confusion matrix.
    test_dataset (Dataset): dataset in the SegmentationDataset format
    model (nn.Module): model to evaluate
    use_correction (bool): if true, corrections are cleaned with the clean algorithm
    scale (int): dimension of confusion matrix
    confusion (bool): if true, computes confusion matrix (default: True)
    report (bool): if true, computes classification report (default: True)
    """
    plt.rcParams["figure.figsize"] = (2*scale, 1*scale)
    plt.rcParams['font.size'] = 1.7*scale

    preds = []
    labs = []
    labs_names = test_dataset.labels_list

    for i, sample in enumerate(test_dataset):

        # True label
        labs.extend(sample['labels'])

        # Model Prediction
        x = torch.Tensor(sample['embeds']).unsqueeze(0)

        pred = model.predict(x).squeeze(0).tolist()

        if use_correction:
            pred = clean_prediction(pred)

        preds.extend(pred)

    # Report
    if report:
        print(classification_report(labs, preds, target_names=labs_names))

    # Confusion Matrix
    if confusion:
        cm = confusion_matrix(labs, preds)
        disp = ConfusionMatrixDisplay(cm, display_labels=labs_names)
        disp.plot()
        plt.show()


def plot_train_eval(history, plot_best=False, scale=6, min_loss=None, max_loss=None, min_f1=None, max_f1=None):
    """
    Plots the train - eval history of a training phase.
    history (dict): with 'train_loss', 'valid_loss', 'train_f1', 'valid_f1', 'train_pk', 'valid_pk', 'best_epoch'
    plot_best (bool): if true, is signed on the plots the best epoch
    scale (int): size of the plots
    min_loss (float): minimum value for loss plot
    min_loss (float): maximum value for loss plot
    min_f1 (float): minimum value for f1 plot
    min_f1 (float): maximum value for f1 plot
    """
    plt.rcParams["figure.figsize"] = (3 * scale, 1 * scale)
    plt.rcParams['font.size'] = 2.0 * scale

    fig, axs = plt.subplots(1, 3)

    x = np.array([i for i in range(len(history['train_loss']))])
    t_loss = np.array(history['train_loss'])
    v_loss = np.array(history['valid_loss'])
    t_f1 = np.array(history['train_f1'])
    v_f1 = np.array(history['valid_f1'])
    t_pk = np.array(history['train_pk'])
    v_pk = np.array(history['valid_pk'])

    # Loss Plot
    axs[0].set_title('LOSS')
    axs[0].plot(x, t_loss)
    axs[0].plot(x, v_loss)
    axs[0].axis(ymin=min_loss, ymax=max_loss)
    axs[0].legend(['train', 'validation'])

    # F1-Score Plot
    axs[1].set_title('F1 SCORE')
    axs[1].plot(x, t_f1)
    axs[1].plot(x, v_f1)
    axs[1].axis(ymin=min_f1, ymax=max_f1)
    axs[1].legend(['train', 'validation'])

    # PK-Score Plot
    axs[2].set_title('PK SCORE')
    axs[2].plot(x, t_pk)
    axs[2].plot(x, v_pk)
    axs[2].axis(ymin=min_f1, ymax=max_f1)
    axs[2].legend(['train', 'validation'])

    # Best Epoch Line
    if plot_best:
        try:
            best_epoch = history['best_epoch']
            y_line = np.array([i for i in range(100)])
            y_line = y_line / 100
            x_line = np.array([best_epoch for i in range(100)])

            for i in range(3):
                axs[i].plot(x_line, y_line, linestyle='dashed')
                axs[i].legend(['train', 'validation', 'best model'])
        except:
            for i in range(3):
                axs[i].legend(['train', 'validation'])