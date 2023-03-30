import torch
import numpy as np
from experiments.model import Kenn
from kenn.boost_functions import GodelBoostConormApprox, LukasiewiczBoostConorm, ProductBoostConorm
import os
import experiments.settings as s
from experiments.util import *
import pandas as pd
from experiments.train_scripts.train_baseline import train_and_evaluate_standard

from experiments.preprocessing import generate_dataset, get_train_and_valid_lengths
import random
from numpy.typing import ArrayLike
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def train_and_evaluate_kenn_inductive(knowldege_path, boost_function=GodelBoostConormApprox, use_preactivations=True, verbose=True):
    """
    Trains KENN model with the Training Set using the Inductive Paradigm.
    Validates on Validation Set, and evaluates accuracy on the Test Set.
    """
    kenn_model = Kenn(knowldege_path, s.NUMBER_OF_FEATURES,
                      boost_function=boost_function,
                      use_preactivations=use_preactivations)

    optimizer = torch.optim.Adam(kenn_model.parameters(), lr=0.01)

    # LOADING DATASET
    features_train = np.genfromtxt(s.DATASET_FOLDER + 'training_features.csv', delimiter=',')
    features_valid = np.genfromtxt(s.DATASET_FOLDER + 'validation_features.csv', delimiter=',')
    features_test = np.genfromtxt(s.DATASET_FOLDER + 'test_features.csv', delimiter=',')
    labels_train = torch.Tensor(np.genfromtxt(s.DATASET_FOLDER + 'training_labels.csv', delimiter=','))
    labels_valid = torch.Tensor(np.genfromtxt(s.DATASET_FOLDER + 'validation_labels.csv', delimiter=','))
    labels_test = torch.Tensor(np.genfromtxt(s.DATASET_FOLDER + 'test_labels.csv', delimiter=','))


    # Import s_x and s_y for the INDUCTIVE learning paradigm
    indexes_training = np.genfromtxt(s.DATASET_FOLDER + 'indexes_training.csv', delimiter=',')
    indexes_valid = np.genfromtxt(s.DATASET_FOLDER + 'indexes_validation.csv', delimiter=',')
    indexes_test = np.genfromtxt(s.DATASET_FOLDER + 'indexes_test.csv', delimiter=',')
    relations_train = np.genfromtxt(s.DATASET_FOLDER + 'relations_training.csv', delimiter=',')
    relations_valid = np.genfromtxt(s.DATASET_FOLDER + 'relations_validation.csv', delimiter=',')
    relations_test = np.genfromtxt(s.DATASET_FOLDER + 'relations_test.csv', delimiter=',')


    index_x_train = torch.tensor(np.expand_dims(indexes_training[:, 0], axis=1), dtype=torch.int64)
    index_y_train = torch.tensor(np.expand_dims(indexes_training[:, 1], axis=1), dtype=torch.int64)
    relations_inductive_training = torch.tensor(relations_train, dtype=torch.float32)
    index_x_valid = torch.tensor(np.expand_dims(indexes_valid[:, 0], axis=1), dtype=torch.int64)
    index_y_valid = torch.tensor(np.expand_dims(indexes_valid[:, 1], axis=1), dtype=torch.int64)
    relations_inductive_valid = torch.tensor(relations_valid, dtype=torch.float32)
    index_x_test = torch.tensor(np.expand_dims(indexes_test[:, 0], axis=1), dtype=torch.int64)
    index_y_test = torch.tensor(np.expand_dims(indexes_test[:, 1], axis=1), dtype=torch.int64)
    relations_inductive_test = torch.tensor(relations_test, dtype=torch.float32)

    print(features_test.shape)
    #train_len, samples_in_valid = get_train_and_valid_lengths(
    #    features, percentage_of_training)

    train_losses = []
    valid_losses = []
    valid_accuracies = []
    train_accuracies = []

    # list of all the evolutions of the clause weights
    clause_weights_1 = []
    # clause_weights_2 = []
    # clause_weights_3 = []

    #train_indices = range(train_len)
    #valid_indices = range(train_len, train_len + samples_in_valid)
    #test_indices = range(train_len + samples_in_valid, features.shape[0])


    # TRAIN AND EVALUATE KENN MODEL
    for epoch in range(s.EPOCHS_KENN):
        train_step_kenn_inductive(
            model=kenn_model,
            features=torch.tensor(features_train, dtype=torch.float32),
            relations=relations_inductive_training,
            index_x_train=index_x_train,
            index_y_train=index_y_train,
            labels=labels_train,
            optimizer=optimizer
        )

        t_predictions = kenn_model(
            [torch.tensor(features_train, dtype=torch.float32), relations_inductive_training, index_x_train, index_y_train])
        t_loss = loss(t_predictions, labels_train)

        # Append current clause weights
        ''' c_enhancers_weights_1 = [float(torch.squeeze(
            ce.conorm_boost.clause_weight)) for ce in kenn_model.kenn_layer_1.binary_ke.clause_enhancers]
        clause_weights_1.append(c_enhancers_weights_1) '''
        # c_enhancers_weights_2 = [float(torch.squeeze(
        #     ce.clause_weight)) for ce in kenn_model.kenn_layer_2.binary_ke.clause_enhancers]
        # clause_weights_2.append(c_enhancers_weights_2)
        # c_enhancers_weights_3 = [float(torch.squeeze(
        #     ce.clause_weight)) for ce in kenn_model.kenn_layer_3.binary_ke.clause_enhancers]
        # clause_weights_3.append(c_enhancers_weights_3)

        v_predictions, v_loss = validation_step_kenn_inductive(
            model=kenn_model,
            features=torch.tensor(features_valid, dtype=torch.float32),
            relations=relations_inductive_valid,
            index_x_valid=index_x_valid,
            index_y_valid=index_y_valid,
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

    predictions_test = kenn_model(
        [torch.tensor(features_test, dtype=torch.float32), relations_inductive_test, index_x_test, index_y_test], save_debug_data=True)

    test_accuracy = accuracy(predictions_test, labels_test.clone().detach()).detach().numpy()

    # all_clause_weights = np.array(
    #     [clause_weights_1])
    print("Test Accuracy: {}".format(test_accuracy))
    return {
        # "train_losses": train_losses,
        "train_accuracies": train_accuracies,
        # "valid_losses": valid_losses,
        "valid_accuracies": valid_accuracies,
        "test_accuracy": test_accuracy}
        # "clause_weights": all_clause_weights,
        # "kenn_test_predictions": predictions_test}


if __name__ == "__main__":
    '''random_seed = 0
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed) '''
    standard_accuracy = []
    kenn_accuracy = []
    impl_accuracy = []
    #generate_dataset(0.75)
    for i in range(50):
        standard_accuracy.append(train_and_evaluate_standard()[3]["test_accuracy"])
        kenn_accuracy.append(train_and_evaluate_kenn_inductive("../knowledge_base")["test_accuracy"])
        impl_accuracy.append(train_and_evaluate_kenn_inductive("../knowledge_implication")["test_accuracy"])

    print("Standard average accuracy", np.mean(standard_accuracy))
    print("kenn average accuracy", np.mean(kenn_accuracy))
    print("impl average accuracy", np.mean(impl_accuracy))
