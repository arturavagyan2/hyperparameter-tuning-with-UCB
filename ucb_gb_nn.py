import random
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier

def ucb_select_hyperparameters(hyperparameter_counts, hyperparameter_rewards, learning_rate_space, n_estimators_space, max_depth_space, hidden_layer_sizes_space=None):
    """
    Select the hyperparameters using the Upper Confidence Bound (UCB) strategy.

    Parameters:
        hyperparameter_counts (ndarray): Array of counts for each combination of hyperparameters.
        hyperparameter_rewards (ndarray): Array of rewards (e.g., accuracy) for each combination of hyperparameters.
        learning_rate_space (list): List of learning rate values.
        n_estimators_space (list): List of n_estimators values.
        max_depth_space (list): List of max_depth values.
        hidden_layer_sizes_space (list, optional): List of hidden_layer_sizes values. Defaults to None.

    Returns:
        tuple: A tuple containing the selected learning rate, n_estimators, max_depth, and optional hidden_layer_sizes.

    Example:
        learning_rate, n_estimators, max_depth, hidden_layer_sizes = ucb_select_hyperparameters(hyperparameter_counts, hyperparameter_rewards,
                                                                                                 learning_rate_space, n_estimators_space,
                                                                                                 max_depth_space, hidden_layer_sizes_space)
    """
    #UCB
    total_counts = np.sum(hyperparameter_counts)
    exploration_factor = 2.0
    epsilon = 1e-6 #to avoid zero division

    ucb_values = hyperparameter_rewards / (hyperparameter_counts + epsilon) + exploration_factor * np.sqrt(np.log(total_counts + 1) / (hyperparameter_counts + epsilon))
    selected_index = np.argmax(ucb_values)

    #variables to return
    num_combinations = len(n_estimators_space) * len(max_depth_space)
    selected_learning_rate = learning_rate_space[selected_index // (num_combinations)]
    selected_n_estimators = n_estimators_space[(selected_index // len(max_depth_space)) % len(n_estimators_space)]
    selected_max_depth = max_depth_space[selected_index % len(max_depth_space)]

    if hidden_layer_sizes_space is not None:
        selected_hidden_layer_sizes = hidden_layer_sizes_space[(selected_index // num_combinations) % len(hidden_layer_sizes_space)]
        return selected_learning_rate, selected_n_estimators, selected_max_depth, selected_hidden_layer_sizes

    return selected_learning_rate, selected_n_estimators, selected_max_depth


def GB_classifier_valerr(X_train, y_train, X_test, y_test, learning_rate_space, n_estimators_space, hyperparameter_counts, 
                         hyperparameter_rewards, num_hyperparameters, max_depth_space):
    """
    Perform gradient boosting classification with hyperparameter selection using UCB and random strategy.

    Parameters:
        X_train (array-like): Training input features.
        y_train (array-like): Training target labels.
        X_test (array-like): Test input features.
        y_test (array-like): Test target labels.
        learning_rate_space (list): List of learning rate values.
        n_estimators_space (list): List of n_estimators values.
        hyperparameter_counts (ndarray): Array of counts for each combination of hyperparameters.
        hyperparameter_rewards (ndarray): Array of rewards (e.g., accuracy) for each combination of hyperparameters.
        num_hyperparameters (int): Number of hyperparameters to evaluate.
        max_depth_space (list): List of max_depth values.

    Returns:
        None

    Example:
        GB_classifier_valerr(X_train, y_train, X_test, y_test, learning_rate_space, n_estimators_space,
                             hyperparameter_counts, hyperparameter_rewards, num_hyperparameters, max_depth_space)
    """
    for _ in range(num_hyperparameters):
        selected_learning_rate_ucb, selected_n_estimators_ucb, selected_max_depth_ucb = ucb_select_hyperparameters(
            hyperparameter_counts, hyperparameter_rewards, learning_rate_space, n_estimators_space, max_depth_space
        )
    
        # Model with UCB-selected hyperparameters
        model_ucb = GradientBoostingClassifier(learning_rate=selected_learning_rate_ucb, 
                                               n_estimators=selected_n_estimators_ucb, max_depth=selected_max_depth_ucb)
        model_ucb.fit(X_train, y_train)
        y_pred_ucb = model_ucb.predict(X_test)
        accuracy_ucb = accuracy_score(y_test, y_pred_ucb)
        validation_error_ucb = 1 - accuracy_ucb

        hyperparameter_index_ucb = learning_rate_space.index(selected_learning_rate_ucb) * len(n_estimators_space) * len(max_depth_space) + n_estimators_space.index(selected_n_estimators_ucb) * len(max_depth_space) + max_depth_space.index(selected_max_depth_ucb)
        hyperparameter_counts[hyperparameter_index_ucb] += 1
        hyperparameter_rewards[hyperparameter_index_ucb] += accuracy_ucb.mean()

    selected_learning_rate_random = random.choice(learning_rate_space)
    selected_n_estimators_random = random.choice(n_estimators_space)
    selected_max_depth_random = random.choice(max_depth_space)
    
    # Model with randomly selected hyperparameters
    model_random = GradientBoostingClassifier(learning_rate=selected_learning_rate_random, n_estimators=selected_n_estimators_random, 
                                                max_depth=selected_max_depth_random)
    model_random.fit(X_train, y_train)
    y_pred_random = model_random.predict(X_test)
    accuracy_random = accuracy_score(y_test, y_pred_random)
    validation_error_random = 1 - accuracy_random

    print(f"Best validation error (UCB): {validation_error_ucb:.6f} with learning rate {selected_learning_rate_ucb}, number of estimators {selected_n_estimators_ucb} and max depth {selected_max_depth_ucb}")
    print(f"Best validation error (Random strategy): {validation_error_random:.6f} with learning rate {selected_learning_rate_random}, number of estimators {selected_n_estimators_random} and max depth {selected_max_depth_random}")

def GB_best_params(X_train, y_train, X_test, y_test, hyperparameter_rewards, learning_rate_space, n_estimators_space, max_depth_space):
    """
    Compare models with the best hyperparameters selected using UCB and random strategy.

    Parameters:
        X_train (array-like): Training input features.
        y_train (array-like): Training target labels.
        X_test (array-like): Test input features.
        y_test (array-like): Test target labels.
        hyperparameter_rewards (ndarray): Array of rewards (e.g., accuracy) for each combination of hyperparameters.
        learning_rate_space (list): List of learning rate values.
        n_estimators_space (list): List of n_estimators values.
        max_depth_space (list): List of max_depth values.

    Returns:
        None

    Example:
        UCB_random_models(X_train, y_train, X_test, y_test, hyperparameter_rewards, learning_rate_space, n_estimators_space, max_depth_space)
    """
    best_hyperparameter_index = np.argmax(hyperparameter_rewards)
    best_learning_rate = learning_rate_space[best_hyperparameter_index // (len(n_estimators_space) * len(max_depth_space))]
    best_n_estimators = n_estimators_space[(best_hyperparameter_index // len(max_depth_space)) % len(n_estimators_space)]
    best_max_depth = max_depth_space[best_hyperparameter_index % len(max_depth_space)]

    learning_rate_random = random.choice(learning_rate_space)
    n_estimators_random = random.choice(n_estimators_space)
    max_depth_random = random.choice(max_depth_space)

    # Model with UCB-selected hyperparameters
    best_model = GradientBoostingClassifier(learning_rate=best_learning_rate, n_estimators=best_n_estimators, max_depth=best_max_depth)
    best_model.fit(X_train, y_train)
    y_pred_best = best_model.predict(X_test)
    accuracy_best = accuracy_score(y_test, y_pred_best)

    # Model with randomly selected hyperparameters
    model_random = GradientBoostingClassifier(learning_rate=learning_rate_random, n_estimators=n_estimators_random, max_depth=max_depth_random)
    model_random.fit(X_train, y_train)
    y_pred_random = model_random.predict(X_test)
    accuracy_random = accuracy_score(y_test, y_pred_random)

    print("Best hyperparameters:")
    print(f"- Learning Rate UCB: {best_learning_rate}")
    print(f"- Learning Rate random: {learning_rate_random}")
    print(f"- Number of Estimators UCB: {best_n_estimators}")
    print(f"- Number of Estimators random: {n_estimators_random}")
    print(f"- Max Depth UCB: {best_max_depth}")
    print(f"- Max Depth random: {max_depth_random}")
    print(f"- Accuracy with UCB hyperparameters: {accuracy_best:.6f}")
    print(f"- Accuracy with random hyperparameters: {accuracy_random:.6f}")

def neural_network_valerr(X_train, y_train, X_test, y_test, num_hyperparameters, hyperparameter_counts, hyperparameter_rewards, 
                       learning_rate_space, n_estimators_space, max_depth_space, hidden_layer_sizes_space):
    """
    Perform neural network classification with hyperparameter selection using UCB and random strategy.

    Parameters:
        X_train (array-like): Training input features.
        y_train (array-like): Training target labels.
        X_test (array-like): Test input features.
        y_test (array-like): Test target labels.
        num_hyperparameters (int): Number of hyperparameters to evaluate.
        hyperparameter_counts (ndarray): Array of counts for each combination of hyperparameters.
        hyperparameter_rewards (ndarray): Array of rewards (e.g., accuracy) for each combination of hyperparameters.
        learning_rate_space (list): List of learning rate values.
        n_estimators_space (list): List of n_estimators values.
        max_depth_space (list): List of max_depth values.
        hidden_layer_sizes_space (list): List of hidden_layer_sizes values.

    Returns:
        None

    Example:
        neural_network_valerr(X_train, y_train, X_test, y_test, num_hyperparameters, hyperparameter_counts, hyperparameter_rewards,
                              learning_rate_space, n_estimators_space, max_depth_space, hidden_layer_sizes_space)
    """
    for _ in range(num_hyperparameters):
        selected_learning_rate_ucb, selected_n_estimators_ucb, selected_max_depth_ucb, selected_hidden_layer_sizes_ucb = ucb_select_hyperparameters(
            hyperparameter_counts, hyperparameter_rewards, learning_rate_space, n_estimators_space, max_depth_space, hidden_layer_sizes_space
        )

        # Model with UCB-selected hyperparameters
        model_ucb_nn = MLPClassifier(learning_rate_init=selected_learning_rate_ucb, hidden_layer_sizes=selected_hidden_layer_sizes_ucb)
        model_ucb_nn.fit(X_train, y_train)
        y_pred_ucb_nn = model_ucb_nn.predict(X_test)
        accuracy_ucb = accuracy_score(y_test, y_pred_ucb_nn)
        validation_error_ucb_nn = 1 - accuracy_ucb

        hyperparameter_index_ucb = learning_rate_space.index(selected_learning_rate_ucb) * len(n_estimators_space) * len(max_depth_space) + \
                                n_estimators_space.index(selected_n_estimators_ucb) * len(max_depth_space) + \
                                max_depth_space.index(selected_max_depth_ucb)
        hyperparameter_counts[hyperparameter_index_ucb] += 1
        hyperparameter_rewards[hyperparameter_index_ucb] += accuracy_ucb
    
    selected_learning_rate_random = random.choice(learning_rate_space)
    selected_n_estimators_random = random.choice(n_estimators_space)
    selected_max_depth_random = random.choice(max_depth_space)
    selected_hidden_layer_sizes_random = random.choice(hidden_layer_sizes_space)

    # Model with randomly selected hyperparameters
    model_random = MLPClassifier(learning_rate_init=selected_learning_rate_random, hidden_layer_sizes=selected_hidden_layer_sizes_random)
    model_random.fit(X_train, y_train)
    y_pred_random = model_random.predict(X_test)
    accuracy_random_nn = accuracy_score(y_test, y_pred_random)
    validation_error_random_nn = 1 - accuracy_random_nn

    hyperparameter_index_random = learning_rate_space.index(selected_learning_rate_random) * len(n_estimators_space) * len(max_depth_space) + \
                                n_estimators_space.index(selected_n_estimators_random) * len(max_depth_space) + \
                                max_depth_space.index(selected_max_depth_random)
    hyperparameter_counts[hyperparameter_index_random] += 1
    hyperparameter_rewards[hyperparameter_index_random] += accuracy_random_nn


    print(f"Best validation error (UCB): {validation_error_ucb_nn:.6f} with learning rate {selected_learning_rate_ucb}, number of estimators {selected_n_estimators_ucb} and max depth {selected_max_depth_ucb}")
    print(f"Best validation error (Random): {validation_error_random_nn:.6f} with learning rate {selected_learning_rate_random}, number of estimators {selected_n_estimators_random} and max depth {selected_max_depth_random}")

def NN_best_params(X_train, y_train, X_test, y_test, hyperparameter_rewards, 
                   learning_rate_space, n_estimators_space, max_depth_space, hidden_layer_sizes_space):
    """
    Perform neural network classification with best hyperparameters selected using UCB and random strategy.

    Parameters:
        X_train (array-like): Training input features.
        y_train (array-like): Training target labels.
        X_test (array-like): Test input features.
        y_test (array-like): Test target labels.
        hyperparameter_rewards (ndarray): Array of rewards (e.g., accuracy) for each combination of hyperparameters.
        learning_rate_space (list): List of learning rate values.
        n_estimators_space (list): List of n_estimators values.
        max_depth_space (list): List of max_depth values.
        hidden_layer_sizes_space (list): List of hidden_layer_sizes values.

    Returns:
        None

    Example:
        NN_best_params(X_train, y_train, X_test, y_test, hyperparameter_rewards, learning_rate_space, n_estimators_space,
                       max_depth_space, hidden_layer_sizes_space)
    """
    best_hyperparameter_index = np.argmax(hyperparameter_rewards)
    best_learning_rate = learning_rate_space[best_hyperparameter_index // (len(n_estimators_space) * len(max_depth_space))]
    best_hidden_layer_sizes = hidden_layer_sizes_space[best_hyperparameter_index % len(hidden_layer_sizes_space)]

    learning_rate_random = random.choice(learning_rate_space)
    hidden_layer_sizes_random = random.choice(hidden_layer_sizes_space)

    # Model with best UCB-selected hyperparameters
    best_model = MLPClassifier(learning_rate_init=best_learning_rate, hidden_layer_sizes=best_hidden_layer_sizes)
    best_model.fit(X_train, y_train)
    y_pred_best = best_model.predict(X_test)
    accuracy_best = accuracy_score(y_test, y_pred_best)

    # Model with randomly selected hyperparameters
    model_random = MLPClassifier(learning_rate_init=learning_rate_random, hidden_layer_sizes=hidden_layer_sizes_random)
    model_random.fit(X_train, y_train)
    y_pred_random = model_random.predict(X_test)
    accuracy_random = accuracy_score(y_test, y_pred_random)

    print("Best hyperparameters:")
    print(f"- Learning Rate (UCB): {best_learning_rate}")
    print(f"- Learning Rate (Random): {learning_rate_random}")
    print(f"- Hidden Layer Sizes (UCB): {best_hidden_layer_sizes}")
    print(f"- Hidden Layer Sizes (Random): {hidden_layer_sizes_random}")
    print(f"- Accuracy with UCB-selected hyperparameters: {accuracy_best:.6f}")
    print(f"- Accuracy with randomly selected hyperparameters: {accuracy_random:.6f}")