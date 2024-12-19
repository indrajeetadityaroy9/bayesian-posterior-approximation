import matplotlib.pyplot as plt
import numpy as np


def plot1(x, labels):
    # Set default font sizes for various elements of the plot
    plt.rc('font', size=12)  # Default text sizes
    plt.rc('axes', titlesize=12)  # Fontsize of the axes titles
    plt.rc('axes', labelsize=12)  # Fontsize of the x and y labels
    plt.rc('xtick', labelsize=10)  # Fontsize of the tick labels
    plt.rc('ytick', labelsize=10)  # Fontsize of the tick labels
    plt.rc('legend', fontsize=10)  # Legend fontsize
    plt.rc('figure', titlesize=14)  # Fontsize of the figure title

    # Define different markers for different classes
    markers = ['o', '^', 's', 'x']
    fig, ax = plt.subplots(figsize=(8, 8))  # Create a figure and a set of subplots

    # Plot each class with a unique marker
    for i in range(len(np.unique(labels))):
        ax.scatter(x[0, labels == i], x[1, labels == i], alpha=0.5, marker=markers[i], label=f"Class {i}")

    # Set labels for the x and y axes
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")

    plt.legend()  # Display legend
    plt.tight_layout()  # Adjust the layout
    plt.show()  # Display the plot


def plot2(n_train, perceptron_count):
    # Create a plot for the number of perceptrons vs. training set size
    fig, ax = plt.subplots()
    ax.plot(n_train, perceptron_count, marker='o', linestyle='-', color='b', label='Number of Perceptrons')

    ax.set_xscale('log')  # Set x-axis to logarithmic scale

    # Set x and y labels and the title of the plot
    ax.set_xlabel('Number of Training Samples')
    ax.set_ylabel('Number of Perceptrons')
    ax.set_title('Number of Perceptrons vs. Training Set Size')

    ax.legend()  # Display legend
    ax.grid(True, which="both", ls="--")  # Enable grid with specific style

    # Annotate each data point with its corresponding value
    for i, txt in enumerate(perceptron_count):
        ax.annotate(f"{txt}", (n_train[i], perceptron_count[i]), textcoords="offset points", xytext=(0, 10),
                    ha='center')

    plt.show()  # Display the plot


def plot3(n_train, error_probs, min_prob_error):
    # Create a plot for error probability vs. number of training samples
    plt.figure(figsize=(10, 6))
    plt.semilogx(n_train, error_probs, marker='o', linestyle='-', color='b')  # Plot with log scale for x-axis

    # Draw a horizontal line representing the theoretical optimal probability of error
    plt.axhline(y=min_prob_error, color="red", linestyle="--", label="Theoretical Optimal P(error)")

    # Annotate each data point with its corresponding probability of error value
    for i, txt in enumerate(error_probs):
        plt.annotate(f"{txt:.4f}", (n_train[i], error_probs[i]), textcoords="offset points", xytext=(0, 10),
                     ha='center')

    # Set x and y labels and the title of the plot
    plt.xlabel('Number of Training Samples')
    plt.ylabel('P(error)')
    plt.title('Error Probability vs. Number of Training Samples')

    plt.legend()  # Display legend
    plt.grid(True, which="both", ls="--")  # Enable grid with specific style
    plt.show()  # Display the plot


def plot4(n_train, error_probs, min_prob_error):
    # Similar to plot3, but specifically for the empirical probability of error of a trained MLP
    plt.figure(figsize=(10, 6))
    plt.semilogx(n_train, error_probs, 'o-', label='Empirical P(error)')

    # Draw a horizontal line for the theoretical optimal probability of error
    plt.axhline(y=min_prob_error, color="red", linestyle="--", label="Theoretical Optimal P(error)")

    # Annotate each data point with its corresponding probability of error value
    for i, txt in enumerate(error_probs):
        plt.annotate(f"{txt:.4f}", (n_train[i], error_probs[i]), textcoords="offset points", xytext=(0, 10),
                     ha='center')

    # Set x and y labels and the title of the plot
    plt.xlabel('Number of Training Samples')
    plt.ylabel('Probability of Error')
    plt.title('Empirical P(error) of Trained MLP vs No. of Training Samples')

    plt.legend()  # Display legend
    plt.grid(True, which="both", ls="--")  # Enable grid with specific style
    plt.show()  # Display the plot