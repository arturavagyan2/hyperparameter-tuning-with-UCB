# Project Name

This project aims to perform hyperparameter selection for a gradient boosting classifier using the Upper Confidence Bound (UCB) algorithm and a random strategy. It explores different combinations of learning rate and number of estimators to find the optimal hyperparameters that yield the highest accuracy.

## Table of Contents

- [Project Description](#project-description)
- [Installation](#installation)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)

## Project Description

In this project, we implement two strategies for hyperparameter selection: UCB and random strategy. The UCB algorithm evaluates the hyperparameters based on their rewards (accuracy) and exploration factor, while the random strategy selects hyperparameters randomly. The project compares the performance of the two strategies by training and evaluating a gradient boosting classifier with the selected hyperparameters.

## Installation

To use this project, follow these steps:

1. Clone the repository: `git clone https://github.com/your_username/your_project.git`
2. Install the required dependencies: `pip install -r requirements.txt`

## Usage

1. Prepare your data in a suitable format.
2. Import the necessary functions from the project files.
3. Set up the learning rate and number of estimators spaces.
4. Initialize the hyperparameter counts and rewards arrays.
5. Call the `GB_classifier_valerr` function, passing in the required arguments: input features, target labels, learning rate space, number of estimators space, hyperparameter counts, hyperparameter rewards, and the number of hyperparameters to evaluate.
6. Call the `UCB_random_models` function, passing in the required arguments: input features, target labels, hyperparameter rewards, learning rate space, and number of estimators space.
7. Review the printed results that show the best hyperparameters and their corresponding accuracy scores for both UCB and random strategy.

## Dependencies

You can install the dependencies using the `requirements.txt` file provided in the repository.

## Contributing

Contributions are welcome! If you have any suggestions or improvements, please submit a pull request.

