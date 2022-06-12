from matplotlib import pyplot as plt


def results_plot_benchmark():
    # Time
    # Accuracy

    names = ["x", "y"]
    d1 = [100, 20]
    d2 = [50, 1000]
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
    ax1.bar(names, d1, facecolor="w", edgecolor="k", hatch=["/", "////"])
    ax1.spines.right.set_visible(False)
    ax1.spines.top.set_visible(False)
    ax1.set_xlabel("Model")
    ax1.set_ylabel("Time")

    ax2.bar(names, d2, facecolor="w", edgecolor="k", hatch=["/", "////"])
    ax2.spines.right.set_visible(False)
    ax2.spines.top.set_visible(False)
    ax2.set_xlabel("Model")
    ax2.set_xlabel("Accuracy")

    plt.suptitle("Benchmark")
    plt.tight_layout()
    plt.show()


def learning_process():
    pass


if __name__ == '__main__':
    results_plot_benchmark()
