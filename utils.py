import numpy as np
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def get_mouse_trial_arrays(mat_file, mouse_ids):
    """
    Extracts trial arrays for each mouse from the given MATLAB file and mouse ids
    """
    mouse_trial_arrays = {}
    for mouse_id in np.unique(mouse_ids):
        mouse_neurons = np.where(mouse_ids[:, 0] == mouse_id)[0]
        trial_array = []
        for stim_condition_idx in range(mat_file['aligned_zscores'].shape[1]):
            per_stim_condition_array = np.zeros((mat_file['aligned_zscores'][mouse_neurons[0], stim_condition_idx].shape[0], 250, len(mouse_neurons)))
            for i, cell in enumerate(mouse_neurons):
                per_stim_condition_array[:, :, i] = mat_file['aligned_zscores'][cell, stim_condition_idx]
            trial_array.append(per_stim_condition_array)
        trial_array = np.concatenate(trial_array, axis=0)
        mouse_trial_arrays[mouse_id] = trial_array
    return mouse_trial_arrays

def get_mouse_stim_labels(mat_file, mouse_ids):
    """
    Extracts stimulus labels for each mouse from the given MATLAB file and mouse ids
    """
    mouse_stim_labels = {}
    for mouse_id in np.unique(mouse_ids):
        mouse_neurons = np.where(mouse_ids[:, 0] == mouse_id)[0]
        stim_labels = []
        for stim_condition_idx in range(mat_file['aligned_zscores'].shape[1]):
            stim_labels += [stim_condition_idx]*mat_file['aligned_zscores'][mouse_neurons[0],stim_condition_idx].shape[0]
        mouse_stim_labels[mouse_id] = np.array(stim_labels)
    return mouse_stim_labels

def plot_hyperplane(coef, intercept, min_x, max_x, linestyle, label):
    """
    Plots the decision boundary (hyperplane) for a linear classifier.
    """
    # get the separating hyperplane
    a = -coef[0] / coef[1]
    xx = np.linspace(min_x - 5, max_x + 5)  # make sure the line is long enough
    yy = a * xx - (intercept) / coef[1]
    plt.plot(xx, yy, linestyle, label=label)

def plot_pca_decision_boundary(stim_labels, trial_array, stim_condition1, stim_condition2, stim_label1, stim_label2, color_map):
    """
    Plots the PCA and decision boundary between two stimulus conditions.
    Stim labels: stimulus labels of the trials
    Trial array: of shape (trial, time, neuron)
    stim_condition1: idx of the first stimulus condition
    stim_condition2: idx of the second stimulus condition
    stim_label1: label for the first stimulus condition
    stim_label2: label for the second stimulus condition
    color_map: dictionary mapping stimulus conditions to colors
    """
    #trial_array_compare is for comparing two stimulus conditions only
    trial_array_compare = trial_array[np.isin(stim_labels, [stim_condition1, stim_condition2])]
    trial_pop_array_compare = trial_array_compare.reshape(trial_array_compare.shape[0], -1)
    stim_labels_compare = stim_labels[np.isin(stim_labels, [stim_condition1, stim_condition2])]
    X_train, X_test, y_train, y_test = train_test_split(trial_pop_array_compare, stim_labels_compare, test_size=0.1)
    svc = LinearSVC()
    svc.fit(X_train, y_train)
    print("Classification score: ", svc.score(X_test, y_test))
    #now do PCA on trial_pop_array_compare
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(trial_pop_array_compare)
    #Plot condition 1
    plt.scatter(pca_result[stim_labels_compare == stim_condition1, 0], pca_result[stim_labels_compare == stim_condition1, 1], c=color_map[stim_condition1], label=stim_label1)
    #Plot condition 2
    plt.scatter(pca_result[stim_labels_compare == stim_condition2, 0], pca_result[stim_labels_compare == stim_condition2, 1], c=color_map[stim_condition2], label=stim_label2)
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plot_hyperplane(svc.coef_[0] @ pca.components_.T, svc.intercept_, np.min(pca_result[:, 0]) - 10, np.max(pca_result[:, 0]) + 10, 'k--', 'Decision Boundary')
    plt.legend()

def svm_simulations(trial_array, stim_labels, task, simulation_trials=1000, neuron_subset=1):
    """
    trial_array: should be of shape (n_trials, n_timepoints, n_neurons) or (n_trials, n_timepoints * n_neurons) if neuron_subset = 1
    stim_labels: should be of shape (n_trials,) the stimulus labels as integers
    task: a string representing the task being performed, to be used to label the graph later
    simulation_trials: the number of simulation trials to run
    neuron_subset: the fraction of neurons to include in the analysis (default is 1, meaning all neurons are included)
    """
    accuracies = []
    tasks = []
    shuffleds = []
    for simulation_idx in range(simulation_trials):
        if neuron_subset < 1:
            # Randomly select a subset of neurons
            num_neurons = int(trial_array.shape[-1] * neuron_subset)
            selected_neurons = np.random.choice(trial_array.shape[-1], num_neurons, replace=False)
            trial_array_subset = trial_array[:, :, selected_neurons]
            trial_pop_array_subset = trial_array_subset.reshape(trial_array_subset.shape[0], -1)
        else:
            trial_pop_array_subset = trial_array.reshape(trial_array.shape[0], -1)
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(trial_pop_array_subset, stim_labels, test_size=0.2)

        # Create a LinearSVC model
        svm = LinearSVC()
        svm.fit(X_train, y_train)
        accuracies.append(svm.score(X_test, y_test))
        tasks.append(task)
        shuffleds.append(False)
        
        # Shuffle the labels and repeat
        shuffled_labels = np.random.permutation(stim_labels)
        X_train, X_test, y_train, y_test = train_test_split(trial_pop_array_subset, shuffled_labels, test_size=0.2)
        svm.fit(X_train, y_train)
        accuracies.append(svm.score(X_test, y_test))
        tasks.append(task)
        shuffleds.append(True)
    return accuracies, tasks, shuffleds

