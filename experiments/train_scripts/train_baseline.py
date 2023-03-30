from experiments.model import Standard
import numpy as np
import experiments.settings as s
from experiments.preprocessing import *
from experiments.util import *


def train_and_evaluate_standard(verbose=True):
    """
    Trains Standard model with the Training Set, validates on Validation Set
    and evaluates accuracy on the Test Set.
    """
    standard_model = Standard(s.NUMBER_OF_FEATURES)
    optimizer = torch.optim.Adam(standard_model.parameters(), lr=0.01)

    # LOADING DATASET
    features_train = np.genfromtxt(s.DATASET_FOLDER + 'training_features.csv', delimiter=',')
    features_valid = np.genfromtxt(s.DATASET_FOLDER + 'validation_features.csv', delimiter=',')
    features_test = np.genfromtxt(s.DATASET_FOLDER + 'test_features.csv', delimiter=',')
    labels_train = torch.Tensor(np.genfromtxt(s.DATASET_FOLDER + 'training_labels.csv', delimiter=','))
    labels_valid = torch.Tensor(np.genfromtxt(s.DATASET_FOLDER + 'validation_labels.csv', delimiter=','))
    labels_test = torch.Tensor(np.genfromtxt(s.DATASET_FOLDER + 'test_labels.csv', delimiter=','))

    train_losses = []
    valid_losses = []
    valid_accuracies = []
    train_accuracies = []

    #train_indices = range(train_len)
    #valid_indices = range(train_len, train_len + samples_in_valid)
    #test_indices = range(train_len + samples_in_valid, features.shape[0])

    # TRAIN AND EVALUATE STANDARD MODEL
    for epoch in range(s.EPOCHS):
        train_step_standard(
            model=standard_model,
            features=torch.tensor(features_train, dtype=torch.float32),
            labels=labels_train,
            optimizer=optimizer
        )

        _, t_predictions = standard_model(torch.tensor(features_train, dtype=torch.float32))
        t_loss = loss(t_predictions, labels_train)

        v_predictions, v_loss = validation_step_standard(
            model=standard_model,
            features=torch.tensor(features_valid, dtype=torch.float32),
            labels=labels_valid,
        )

        train_losses.append(t_loss)
        valid_losses.append(v_loss)

        t_accuracy = accuracy(t_predictions, labels_train).detach().numpy()
        v_accuracy = accuracy(v_predictions, labels_valid).detach().numpy()

        train_accuracies.append(t_accuracy)
        valid_accuracies.append(v_accuracy)

        if verbose and epoch % 10 == 0:
            print(
                "Epoch {}: Training Loss: {:5.4f} Validation Loss: {:5.4f} | Train Accuracy: {:5.4f} Validation Accuracy: {:5.4f};".format(
                    epoch, t_loss, v_loss, t_accuracy, v_accuracy))

        # Early Stopping
        stopEarly = callback_early_stopping(valid_accuracies)
        if stopEarly:
            print("callback_early_stopping signal received at epoch= %d/%d" %
                  (epoch, s.EPOCHS))
            print("Terminating training ")
            break

    preactivations_train, _ = standard_model(torch.tensor(features_train, dtype=torch.float32))
    preactivations_valid, _ = standard_model(torch.tensor(features_valid, dtype=torch.float32))
    preactivations_test, predictions_test = standard_model(
        torch.tensor(features_test, dtype=torch.float32))
    test_accuracy = accuracy(predictions_test, labels_test).detach().numpy()
    print("Test Accuracy: {}".format(test_accuracy))

    nn_results = {
        # "train_losses": train_losses,
                  "train_accuracies": train_accuracies,
                  # "valid_losses": valid_losses,
                  "valid_accuracies": valid_accuracies,
                  "test_accuracy": test_accuracy}

    return (
        preactivations_train,
        preactivations_valid,
        preactivations_test,
        nn_results
    )