import random
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier

def ucb_select_hyperparameters(hyperparameter_counts, hyperparameter_rewards, learning_rate_space, n_estimators_space):
    """
    Select the hyperparameters using the Upper Confidence Bound (UCB) strategy.

    Parameters:
        hyperparameter_counts (ndarray): Array of counts for each combination of hyperparameters.
        hyperparameter_rewards (ndarray): Array of rewards (e.g., accuracy) for each combination of hyperparameters.
        learning_rate_space (list): List of learning rate values.
        n_estimators_space (list): List of n_estimators values.

    Returns:
        tuple: A tuple containing the selected learning rate and n_estimators.

    Example:
        learning_rate, n_estimators = ucb_select_hyperparameters(hyperparameter_counts, hyperparameter_rewards,
                                                                 learning_rate_space, n_estimators_space)
    """
    total_counts = np.sum(hyperparameter_counts)
    exploration_factor = 2.0
    ucb_values = hyperparameter_rewards / (hyperparameter_counts + 1e-6) + exploration_factor * np.sqrt(np.log(total_counts + 1) / (hyperparameter_counts + 1e-6))
    selected_index = np.argmax(ucb_values)
    selected_learning_rate = learning_rate_space[selected_index // len(n_estimators_space)]
    selected_n_estimators = n_estimators_space[selected_index % len(n_estimators_space)]
    
    return selected_learning_rate, selected_n_estimators

def GB_classifier_valerr(X, y, learning_rate_space, n_estimators_space, hyperparameter_counts, hyperparameter_rewards, num_hyperparameters):
    """
    Perform gradient boosting classification with hyperparameter selection using UCB and random strategy.

    Parameters:
        X (array-like): Input features.
        y (array-like): Target labels.
        learning_rate_space (list): List of learning rate values.
        n_estimators_space (list): List of n_estimators values.
        hyperparameter_counts (ndarray): Array of counts for each combination of hyperparameters.
        hyperparameter_rewards (ndarray): Array of rewards (e.g., accuracy) for each combination of hyperparameters.
        num_hyperparameters (int): Number of hyperparameters to evaluate.

    Returns:
        None

    Example:
        GB_classifier_valerr(X_train, y_train, learning_rate_space, n_estimators_space,
                             hyperparameter_counts, hyperparameter_rewards, num_hyperparameters)
    """
    for _ in range(num_hyperparameters):
        selected_learning_rate_ucb, selected_n_estimators_ucb = ucb_select_hyperparameters(
            hyperparameter_counts, hyperparameter_rewards, learning_rate_space, n_estimators_space
        )
     
        selected_learning_rate_random = random.choice(learning_rate_space)
        selected_n_estimators_random = random.choice(n_estimators_space)
        
        #model with UCB
        model_ucb = GradientBoostingClassifier(learning_rate=selected_learning_rate_ucb, n_estimators=selected_n_estimators_ucb)
        model_ucb.fit(X, y)
        y_pred_ucb = model_ucb.predict(X)
        accuracy_ucb = accuracy_score(y, y_pred_ucb)
        validation_error_ucb = 1 - accuracy_ucb

        hyperparameter_index_ucb = learning_rate_space.index(selected_learning_rate_ucb) * len(n_estimators_space) + n_estimators_space.index(selected_n_estimators_ucb)
        hyperparameter_counts[hyperparameter_index_ucb] += 1
        hyperparameter_rewards[hyperparameter_index_ucb] += accuracy_ucb

        #model with random parameters
        model_random = GradientBoostingClassifier(learning_rate=selected_learning_rate_random, n_estimators=selected_n_estimators_random)
        model_random.fit(X, y)
        y_pred_random = model_random.predict(X)
        accuracy_random = accuracy_score(y, y_pred_random)
        validation_error_random = 1 - accuracy_random

        hyperparameter_index_random = learning_rate_space.index(selected_learning_rate_random) * len(n_estimators_space) + n_estimators_space.index(selected_n_estimators_random)
        hyperparameter_counts[hyperparameter_index_random] += 1
        hyperparameter_rewards[hyperparameter_index_random] += accuracy_random

    print(f"Best validation error (UCB): {validation_error_ucb} with learning rate {selected_learning_rate_ucb} and number of estimators {selected_n_estimators_ucb}")
    print(f"Best validation error (Random strategy): {validation_error_random} with learning rate {selected_learning_rate_random} and number of estimators {selected_n_estimators_random}")

def UCB_random_models(X, y, hyperparameter_rewards, learning_rate_space, n_estimators_space):
    """
    Compare models with the best hyperparameters selected using UCB and random strategy.

    Parameters:
        X (array-like): Input features.
        y (array-like): Target labels.
        hyperparameter_rewards (ndarray): Array of rewards (e.g., accuracy) for each combination of hyperparameters.
        learning_rate_space (list): List of learning rate values.
        n_estimators_space (list): List of n_estimators values.

    Returns:
        None

    Example:
        UCB_random_models(X_test, y_test, hyperparameter_rewards, learning_rate_space, n_estimators_space)
    """
    best_hyperparameter_index = np.argmax(hyperparameter_rewards)
    best_learning_rate = learning_rate_space[best_hyperparameter_index // len(n_estimators_space)]
    best_n_estimators = n_estimators_space[best_hyperparameter_index % len(n_estimators_space)]

    learning_rate_random = random.choice(learning_rate_space)
    n_estimators_random = random.choice(n_estimators_space)

    #model with UCB
    best_model = GradientBoostingClassifier(learning_rate=best_learning_rate, n_estimators=best_n_estimators)
    best_model.fit(X, y)
    y_pred_best = best_model.predict(X)
    accuracy_best = accuracy_score(y, y_pred_best)

    #model with random parameters
    model_random = GradientBoostingClassifier(learning_rate=learning_rate_random, n_estimators=n_estimators_random)
    model_random.fit(X, y)
    y_pred_random = model_random.predict(X)
    accuracy_random = accuracy_score(y, y_pred_random)

    print("Best hyperparameters:")
    print(f"- Learning Rate UCB: {best_learning_rate}")
    print(f"- Learning Rate random: {learning_rate_random}")
    print(f"- Number of Estimators UCB: {best_n_estimators}")
    print(f"- Number of Estimators random: {n_estimators_random}")
    print(f"- Accuracy with UCB hyperparameters: {accuracy_best:.4f}")
    print(f"- Accuracy with random hyperparameters: {accuracy_random:.4f}")

