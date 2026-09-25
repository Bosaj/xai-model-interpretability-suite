import sys

sys.path.insert(0, ".")

import itertools

import matplotlib.pyplot as plt

# We can keep this import, or rely on passed objects effectively duck-typing
# from Dataset.dataset import Dataset
import numpy as np


def get_grid(model, dataset, points_per_feature=50):
    """
    Retrieve grid data for plotting a two-dimensional graph with `points_per_feature` for each axis.
    The space is created by the hyperparameters' lower and upper values. Only the first two input
    labels are used.

    Parameters:
        model: Classifier which can call a predict method.
        dataset (utils.Dataset): Dataset with access to configspace and input labels.
        points_per_feature: How many points in each dimension.

    Returns:
        u (np.ndarray): x-axis points with shape (`points_per_feature`,).
        v (np.ndarray): y-axis points with shape (`points_per_feature`,).
        z (np.ndarray): Values with shape (`points_per_feature`, `points_per_feature`).
    """

    # hps = dataset.get_configspace().get_hyperparameters_dict()
    # labels = dataset.get_input_labels()

    # Use dataset.X directly to get min/max
    X_data = dataset.X

    # Assuming the first two features in dataset.X correspond to the axes we want
    x1_min, x1_max = np.min(X_data[:, 0]), np.max(X_data[:, 0])
    x2_min, x2_max = np.min(X_data[:, 1]), np.max(X_data[:, 1])

    x1 = np.linspace(x1_min, x1_max, points_per_feature)
    x2 = np.linspace(x2_min, x2_max, points_per_feature)

    X = []
    for x in itertools.product(x1, x2):
        X.append(x)

    X = np.array(X)
    y = model.predict(X)

    # Reshape all x
    u = x1
    v = x2
    z = y.reshape(points_per_feature, points_per_feature).T

    return u, v, z


def plot_grid(u, v, z, labels, title=None, embedded=False):
    """
    Uses the grid data to add a color grid to the plot.

    Parameters:
        u (np.ndarray): x-axis points with shape (N,).
        v (np.ndarray): y-axis points with shape (N,).
        z (np.ndarray): Color values with shape (N, N).
        labels (list): Labels for x and y axis.
        embedded (bool): Whether a new figure should be created or not.

    Returns:
        plt (matplotlib.pyplot or utils.styled_plot.plt): Plot with applied color grid.
    """

    if not embedded:
        plt.figure(figsize=(10, 6))

    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.title(title)

    plt.pcolormesh(u, v, z, cmap="viridis", shading="auto", alpha=0.6)
    plt.colorbar()
    # plt.grid(alpha=0)

    return plt


def plot_points_in_grid(plt, Z=None, y=None, weights=None, colors=None, x_interest=None, size=20):
    """
    Given a plot, add scatter points from `Z` and `x_interest`.

    Parameters:
        plt (matplotlib.pyplot or utils.styled_plot.plt): Plot with color grid.
        Z (np.ndarray): Points with shape (?, 2) which should be added to the plot.
        y (np.ndarray): Target values with shape (?,), of the points added to the plot, determines the colouring of points.
        weights (np.ndarray): Normalized weights with shape (?,), determine the size of points in the plot.
        colors (dict): Returns the color for an y value.
        x_interest (np.ndarray): Single point with shape (2,) whose prediciton we want to explain. If None (default) no point is added.
        size (int): Default size of the markers/points. Default is 8.
    """

    if colors is None:
        colors = {}
    if y is None:
        y = []
    if Z is None:
        Z = []

    unique_y = list(set(y))
    # Default colors if not provided
    default_colors = ["blue", "orange", "green", "red"]

    for i, y_ in enumerate(unique_y):
        idx = np.where(y == y_)[0]

        color = default_colors[i % len(default_colors)]
        if y_ in colors:
            color = colors[y_]

        s = size
        if weights is not None:
            # Scale size by weight
            s = weights[idx] * size * 5  # Amplify weight effect

        plt.scatter(Z[idx, 0], Z[idx, 1], c=color, s=s, label=str(y_), edgecolor="k", alpha=0.8)

    if x_interest is not None:
        plt.scatter(
            [x_interest[0]], [x_interest[1]], c="red", s=size * 2, marker="X", label="Interest Point", zorder=10
        )

    plt.legend()


def sample_points(model, dataset, num_points, seed=0):
    """
    Samples points for the two first features. Uses the bounds from configspace again.

    Parameters:
        model: Classifier which can call a predict method.
        dataset (utils.Dataset): Dataset with access to configspace and input labels.
        num_points (int): How many points should be sampled.
        seed (int): Seed to feed random.

    Returns:
        X (np.ndarray): Data with shape (`num_points`, 2)
        y (np.ndarray): Target values with shape (`num_points`,)
    """
    np.random.seed(seed)
    # hps = dataset.get_configspace().get_hyperparameters_dict()
    # labels = dataset.get_input_labels()

    # Use dataset.X directly
    X_data = dataset.X
    x1_min, x1_max = np.min(X_data[:, 0]), np.max(X_data[:, 0])
    x2_min, x2_max = np.min(X_data[:, 1]), np.max(X_data[:, 1])

    X_sampled = []
    for _ in range(num_points):
        x1 = np.random.uniform(x1_min, x1_max)
        x2 = np.random.uniform(x2_min, x2_max)
        X_sampled.append([x1, x2])

    X_sampled = np.array(X_sampled)
    y_sampled = model.predict(X_sampled)
    return X_sampled, y_sampled


def weight_points(x_interest, Z, kernel_width=0.2):
    """
    For every z in `Z` returns a weight depending on the distance to `x_interest`.

    Parameters:
        x_interest (np.ndarray): Single point with shape (2,) whose prediction we want to explain.
        Z (np.ndarray): Points with shape (?, 2), data which needs to be weighted.
        kernel_width (float): kernel_width value to calculate distance according to exponential kernel.

    Returns:
        weights (np.ndarray): Normalized weights between 0..1 with shape (?,).
    """
    # Euclidean distance
    # Ensure x_interest is 1D array for broadcasting or same shape logic
    # Z is (N, 2), x_interest is (2,)
    d = np.linalg.norm(Z - x_interest, axis=1)

    # Exponential kernel: w = sqrt(exp(-d^2 / width^2))
    # Often just exp(-d^2 / width^2) is used, but following prompt hint from earlier step 21
    weights = np.sqrt(np.exp(-(d**2) / (kernel_width**2)))

    return weights


def fit_explainer_model(Z, y, weights=None, seed=0):
    """
    Fits a decision tree.

    Parameters:
        Z (np.ndarray): Points with shape (?, 2), used to fit surrogate model.
        y (np.ndarray): Target values with shape (?,).
        weights (np.ndarray): Normalized weights with shape (?,).
        seed (int): Seed for the decision tree.

    Returns:
        model (DecisionTreeRegressor): Fitted explainer model.
        # Note: LIME usually uses Regression (Ridge) or a simple Tree.
        # If y is class labels, DecisionTreeClassifier might be better,
        # but the prompt/template implies returning a general model.
        # DecisionTreeRegressor is robust if y is probabilities, or 0/1.
        # Let's use DecisionTreeClassifier if y is discrete (int), else Regressor.
    """
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

    # Check if classification or regression
    # If y is float (probabilities), regressor. If int (classes), classifier.
    # Our usage in notebook passes class labels (0/1).

    if np.issubdtype(y.dtype, np.integer) or len(np.unique(y)) <= 10:
        model = DecisionTreeClassifier(max_depth=3, random_state=seed)
    else:
        model = DecisionTreeRegressor(max_depth=3, random_state=seed)

    model.fit(Z, y, sample_weight=weights)
    return model
