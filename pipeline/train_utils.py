import wandb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix
from timeit import default_timer as timer
from IPython.display import clear_output
from tqdm.notebook import tqdm

from constants import N_CLASSES, device
from criterions import SCORE_MATRIX

############################################################
# Confusion matrices
############################################################


def get_confusion_matrix(predicted_labels, true_labels):
    error_matrix = confusion_matrix(true_labels, predicted_labels, normalize='true')
    return pd.DataFrame(error_matrix, index=[f'true_{i}' for i in range(N_CLASSES)], columns=[f'pred_{i}' for i in range(N_CLASSES)])


def get_score_matrix(error_matrix: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(SCORE_MATRIX * error_matrix.values, index=error_matrix.index, columns=error_matrix.columns)


def get_score(error_matrix: pd.DataFrame) -> float:
    return get_score_matrix(error_matrix).sum().sum()


def plot_confusion_matrix(matrix: pd.DataFrame, use_pct: bool = True):
    FONT_SIZE = 15
    ax = sns.heatmap(matrix, annot=True, fmt=('.1%' if use_pct else '.2f'), cmap='Blues', cbar=False, square=True)
    ax.set_xlabel('Predicted', fontsize=FONT_SIZE)
    ax.set_ylabel('True', fontsize=FONT_SIZE)
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    ax.xaxis.set_tick_params(labeltop=True)  # Set the x-axis labels on top
    ax.yaxis.set_ticklabels(list(range(N_CLASSES)))
    ax.xaxis.set_ticklabels(list(range(N_CLASSES)), ha='center')


def get_accuracy_by_classes(matrices: list[pd.DataFrame]) -> list[list[float]]:
    result = []
    for i in range(N_CLASSES):
        accuracies = [matrix.iloc[i, i] for matrix in matrices]
        result.append(accuracies)
    return result

############################################################
# Train and Validation loops
############################################################


def train_loop(model, optimizer, criterion, train_loader) -> tuple:
    train_loss, train_accuracy = 0.0, 0.0
    predicted_labels = []
    true_labels = []

    model.train()

    for images, labels in tqdm(train_loader, desc='Training'):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        # images: batch_size x num_channels x height x width
        logits = model(images)
        # logits: batch_size x num_classes
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        # save predictions and true labels
        predicted_labels.append(logits.argmax(dim=1).detach().cpu())
        true_labels.append(labels.cpu())

        train_loss += loss.item() * images.shape[0]
        train_accuracy += (logits.detach().argmax(dim=1) == labels).sum().item()

    train_loss /= len(train_loader.dataset)
    train_accuracy /= len(train_loader.dataset)

    predicted_labels = torch.concat(predicted_labels)
    true_labels = torch.concat(true_labels)
    matrix = get_confusion_matrix(predicted_labels, true_labels)

    return train_loss, train_accuracy, matrix


@torch.no_grad()
def val_loop(model, criterion, val_loader) -> tuple:
    val_loss, val_accuracy = 0.0, 0.0
    predicted_labels = []
    true_labels = []

    model.eval()

    for images, labels in tqdm(val_loader, desc='Validating'):
        images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            logits = model(images)
            # logits: batch_size x num_classes
            loss = criterion(logits, labels)

        # save predictions and true labels
        predicted_labels.append(logits.argmax(dim=1).cpu())
        true_labels.append(labels.cpu())

        val_loss += loss.item() * images.shape[0]
        val_accuracy += (logits.argmax(dim=1) == labels).sum().item()

    val_loss /= len(val_loader.dataset)
    val_accuracy /= len(val_loader.dataset)

    predicted_labels = torch.concat(predicted_labels)
    true_labels = torch.concat(true_labels)
    matrix = get_confusion_matrix(predicted_labels, true_labels)

    return val_loss, val_accuracy, matrix

############################################################
# Plotting results
############################################################


def plot_results(lrs, train_losses, val_losses, train_accuracies, val_accuracies, train_scores, val_scores, confusion_matrices_train, confusion_matrices_val):
    clear_output()
    assert len(train_losses) == len(val_losses) == len(train_accuracies) == len(val_accuracies) == len(lrs) == len(train_scores) == len(val_scores)

    print(f'Train accuracy: {train_accuracies[-1]:.1%}. Val accuracy: {val_accuracies[-1]:.1%}')
    print(f'Train score: {train_scores[-1]:.2f}. Val score: {val_scores[-1]:.2f}')
    print(f'Train loss: {train_losses[-1]:.2f}. Val loss: {val_losses[-1]:.2f}')

    # plot losses
    fig, axs = plt.subplots(1, 2, figsize=(13, 4))
    x_epochs = list(range(1, len(train_losses) + 1))
    axs[0].plot(x_epochs, train_losses, label='train')
    axs[0].plot(x_epochs, val_losses, label='val')
    axs[0].set_ylabel('loss')
    axs[0].set_title('Loss')

    axs[1].plot(x_epochs, train_accuracies, label='train')
    axs[1].plot(x_epochs, val_accuracies, label='val')
    axs[1].set_ylabel('accuracy')
    axs[1].set_title('Accuracy')

    for ax in axs:
        ax.set_xlabel('epoch')
        ax.legend(loc='upper left')
    plt.show()

    # plot scores and lr
    fig, axs = plt.subplots(1, 2, figsize=(13, 4))
    axs[0].axhline(0.0, color='black', linestyle='dashed')
    axs[0].plot(x_epochs, train_scores, label='train')
    axs[0].plot(x_epochs, val_scores, label='val')
    axs[0].set_ylabel('score')
    axs[0].set_title('Score')
    axs[0].legend(loc='upper left')

    axs[1].plot(x_epochs, lrs)
    axs[1].set_ylabel('lr')
    axs[1].set_title('LR')

    plt.show()

    # plot accuracy for each class
    fig, axs = plt.subplots(1, 2, figsize=(13, 4))
    for i, (title, confusion_matrices) in enumerate(zip(['Train', 'Val'], [confusion_matrices_train, confusion_matrices_val])):
        axs[i].set_title(title)
        accuracies_by_classes = get_accuracy_by_classes(confusion_matrices)
        for j in range(N_CLASSES):
            axs[i].plot(x_epochs, accuracies_by_classes[j], label=f'{j} class')
        axs[i].set_ylabel('accuracy')
        axs[i].set_xlabel('epoch')
        axs[i].legend(loc='upper left')
    plt.show()

    # plot confusion matrices
    for title, compute_score in zip(['Confusion matrix', 'Score matrix'], [False, True]):
        matrices = [get_score_matrix(matrix) if compute_score else matrix for matrix in [confusion_matrices_train[-1], confusion_matrices_val[-1]]]
        FONT_SIZE = 20
        plt.subplots(1, 2, figsize=(13, 5))

        plt.subplot(1, 2, 1)
        plt.title(f'{title} train', fontsize=FONT_SIZE)
        plot_confusion_matrix(matrices[0], use_pct=not compute_score)

        plt.subplot(1, 2, 2)
        plt.title(f'{title} val',  fontsize=FONT_SIZE)
        plot_confusion_matrix(matrices[1], use_pct=not compute_score)

        plt.show()

############################################################
# Train Model
############################################################


def train_nn(model, optimizer, criterion, scheduler, train_loader, val_loader, n_epochs: int, log_wandb: bool):
    train_losses, train_accuracies, val_losses, val_accuracies = [], [], [], []
    confusion_matrices_train, confusion_matrices_val = [], []
    train_scores, val_scores = [], []
    lrs = []

    for epoch in range(1, n_epochs + 1):
        # Train
        start_time = timer()
        train_loss, train_accuracy, confusion_matrix_train = train_loop(model, optimizer, criterion, train_loader)
        train_time = timer() - start_time

        confusion_matrices_train.append(confusion_matrix_train)
        train_scores.append(get_score(confusion_matrix_train))

        train_losses += [train_loss]
        train_accuracies += [train_accuracy]

        # Validation
        start_time = timer()
        val_loss, val_accuracy, confusion_matrix_val = val_loop(model, criterion, val_loader)
        val_time = timer() - start_time

        confusion_matrices_val.append(confusion_matrix_val)
        val_scores.append(get_score(confusion_matrix_val))

        val_losses += [val_loss]
        val_accuracies += [val_accuracy]

        # Log lr used on current epoch
        lr = next(iter(optimizer.param_groups))['lr']
        lrs.append(lr)

        # Scheduler
        if scheduler is not None:
            scheduler.step(val_loss)

        # Log record
        record = {
            'train_loss': train_loss,
            'val_loss': val_loss,
            'lr': lr,
            'train_time': train_time,
            'val_time': val_time,
        }
        if log_wandb:
            wandb.log(record)

        # Plot
        print(f'Train time: {train_time}. Val time: {val_time}')
        plot_results(lrs, train_losses, val_losses, train_accuracies, val_accuracies, train_scores, val_scores, confusion_matrices_train, confusion_matrices_val)
    if log_wandb:
        wandb.finish()


def main():
    matrix = get_confusion_matrix(np.array([0, 1, 2, 1, 0, 2, 0, 1, 2, 0]), np.array([0, 1, 2, 1, 0, 1, 2, 0, 2, 0]))
    plot_confusion_matrix(matrix)
    print('Accuracies:')
    print(get_accuracy_by_classes([matrix]))
    print('\nConfusion matrix:')
    print(matrix)
    print('\nScore matrix:')
    print(get_score_matrix(matrix))
    print('\nScore:')
    print(get_score(matrix))


if __name__ == '__main__':
    main()
