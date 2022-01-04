import numpy as np
import matplotlib.pyplot as plt

def plot_cv_results(train_loss, val_loss):
    # calculate average training and validation loss
    avg_val_loss = np.mean(val_loss.T, axis=1)
    avg_train_loss = np.mean(train_loss.T, axis=1)

    #
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 5))
    for jj in range(len(train_loss)):
        lab = "fold " + str(jj)
        ax1.semilogy(train_loss[jj], label=lab)
        ax1.set_ylabel("Log loss")
        ax1.set_xlabel("Epoch")
        ax1.set_title("Training loss")
        ax1.legend()

    for jj in range(len(val_loss)):
        lab = "fold " + str(jj)
        ax2.semilogy(val_loss[jj], label=lab)
        ax2.set_ylabel("Log loss")
        ax2.set_xlabel("Epoch")
        ax2.set_title("Hold-out loss")
        ax2.legend()

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(18, 5))
    plt.semilogy(avg_val_loss, label="avg Hold-out loss")
    plt.semilogy(avg_train_loss, label="avg training loss")
    plt.ylabel("Log Loss")
    plt.xlabel("Epoch")
    plt.title("Average Training and Hold-out loss")
    plt.legend()
    plt.show()

def plot_cv_hyper(train_loss, val_loss):
    # calculate average training and validation loss
    avg_val_loss = np.mean(val_loss, axis=1)
    avg_train_loss = np.mean(train_loss, axis=1)

    #
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 5))
    for jj in range(len(avg_train_loss)):
        lab = "hyper comb " + str(jj)
        ax1.semilogy(avg_train_loss[jj], label=lab)
        ax1.set_ylabel("Log loss")
        ax1.set_xlabel("Epoch")
        ax1.set_title("Average Training loss")
        ax1.legend()

    for jj in range(len(avg_val_loss)):
        lab = "hyper comb " + str(jj)
        ax2.semilogy(avg_val_loss[jj], label=lab)
        ax2.set_ylabel("Log loss")
        ax2.set_xlabel("Epoch")
        ax2.set_title("Average Validation loss")
        ax2.legend()

    plt.tight_layout()
    plt.show()


def plot_class_results(train_loss, train_acc, val_loss, val_acc):
    plt.figure(figsize=(18, 5))
    plt.subplot(121)
    plt.semilogy(val_loss, label="Validation loss")
    plt.semilogy(train_loss, label="Training loss")
    plt.ylabel("Log Loss")
    plt.xlabel("Epoch")
    plt.title("Training and Validation loss")
    plt.legend()

    plt.subplot(122)
    plt.plot(val_acc, label="Validation Accuracy")
    plt.plot(train_acc, label="Training Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation accuracy")
    plt.legend()

    plt.show()


def plot_class_hyper(train_loss_hyper, train_acc_hyper, val_loss_hyper, val_acc_hyper):
    plt.figure(figsize=(18, 10))
    plt.subplot(221)
    for j in range(len(train_loss_hyper)):
        lab = "hyper comb" + str(j)
        plt.plot(train_loss_hyper[j], label=lab)
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.title("Training Loss")
    plt.legend()

    plt.subplot(222)
    for j in range(len(val_loss_hyper)):
        lab = "hyper comb" + str(j)
        plt.plot(val_loss_hyper[j], label=lab)
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.title("Validation Loss")
    plt.legend()

    plt.subplot(223)
    for j in range(len(train_acc_hyper)):
        lab = "hyper comb" + str(j)
        plt.plot(train_acc_hyper[j], label=lab)
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.title("Training Accuracy")
    plt.legend()

    plt.subplot(224)
    for j in range(len(val_acc_hyper)):
        lab = "hyper comb" + str(j)
        plt.plot(val_acc_hyper[j], label=lab)
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.title("Validation Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_img_probs(img,target,y_hat,lab_dict):
  plt.subplot(121)
  plt.imshow(img.view(28,28),cmap="Greys")
  plt.title(lab_dict[int(target)])
  plt.subplot(122)
  labs = list(lab_dict.values())
  x = np.arange(len(labs))
  plt.bar(x = x, height = y_hat[0], width = 0.3)
  plt.ylim((0,1))
  plt.xticks(x, labs, rotation = 90)
  plt.ylabel('predicted probability')
  plt.tight_layout()
  plt.show()

def plot_confusion(cm, lab_dict, title='Confusion matrix', cmap=plt.cm.Blues):
    labs = list(lab_dict.values())
    plt.figure(figsize = (7,7))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labs))
    plt.xticks(tick_marks, labs, rotation=90)
    plt.yticks(tick_marks, labs)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')