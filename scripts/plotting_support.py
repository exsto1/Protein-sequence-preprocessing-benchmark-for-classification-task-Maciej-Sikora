from matplotlib import pyplot as plt
import os


def results_plot_benchmark(names, tests, all_times, all_accs, outfile="./presentation/images/benchmark.png"):
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(11, 4))

    for i in range(len(tests)):
        ax1.bar([i0 + 0.15 * (i - 2) for i0 in range(len(names))], all_times[i], width=0.15, facecolor="w",
                edgecolor="k", hatch=["/" * (i + 1)], label=tests[i])
    ax1.spines.right.set_visible(False)
    ax1.spines.top.set_visible(False)
    ax1.set_xticks(range(len(names)), names)
    ax1.set_xlabel("Model", fontsize=12)
    ax1.set_ylabel("Time [sec]", fontsize=12)

    ax1.legend(ncol=2, fontsize="small")

    for i in range(len(tests)):
        ax2.bar([i0 + 0.15 * (i - 2) for i0 in range(len(names))], all_accs[i], width=0.15, facecolor="w",
                edgecolor="k", hatch=["/" * (i + 1)], label=tests[i])
    ax2.spines.right.set_visible(False)
    ax2.spines.top.set_visible(False)
    ax2.set_xticks(range(len(names)), names)
    ax2.set_xlabel("Model", fontsize=12)
    ax2.set_xlabel("Accuracy", fontsize=12)


    plt.suptitle("Comparison of time and accuracy between test", fontsize=18)
    plt.tight_layout()
    plt.savefig(outfile)
    plt.show()


def plot_sizes(files, names, outfile):
    y_data = []
    for file in files:
        st_file = os.stat(file)
        y_data.append(st_file.st_size / (1024 * 1024))

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.bar(names, y_data, facecolor="w", edgecolor="k", hatch="/")
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.set_xlabel("Model", fontsize=11)
    ax.set_ylabel("Size [MB]", fontsize=11)

    plt.suptitle("Comparison of model sizes", fontsize=14)
    plt.tight_layout()
    plt.savefig(outfile)
    plt.show()


if __name__ == '__main__':
    names = ["Original", "Singletons", "Triplets", "Sum\nTriplets", "Sum\nK-mers", "Biovec"]
    tests = ["Decision trees", "Random trees", "Nearest neighbours", "MLP", "Machine Learning"]
    times = [[16.07, 10.66, 6.54, 22.6, 47.2, 0.39], [25.71, 23.79, 25.86, 26.81, 35.63, 5.81],
     [7.27, 4.63, 2.47, 7.25, 7.71, 0.55], [270.7, 169.79, 51.17, 59.72, 174.14, 2.25],
     [64.46, 53.11, 50.17, 60.85, 60.29, 39.75]]
    accs = [[0.1886, 0.7048, 0.6524, 0.8267, 0.3105, 0.7089], [0.419, 0.8019, 0.7876, 0.9781, 0.3419, 0.8839],
     [0.3314, 0.7457, 0.7305, 0.719, 0.301, 0.8712], [0.1914, 0.7781, 0.4362, 0.9952, 0.3495, 0.8376],
     [0.4257, 0.6419, 0.361, 0.9924, 0.3467, 0.7347]]
    results_plot_benchmark(names, tests, times, accs,
                           outfile="../presentation/images/benchmark.png")

    files = ["../data/clean_dataset_original.pkl",
             "../data/clean_dataset_singletons.pkl",
             "../data/clean_dataset_triplets.pkl",
             "../data/clean_dataset_sum_triplets.pkl",
             "../data/clean_dataset_sum_k_mers.pkl",
             "../data/clean_dataset_biovec.pkl"]

    plot_sizes(files, names, "../presentation/images/sizes.png")

    pass
