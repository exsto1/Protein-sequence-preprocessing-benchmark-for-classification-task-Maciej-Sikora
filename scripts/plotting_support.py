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
    ax1.legend()

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

    fig, ax = plt.subplots(figsize=(6, 4))

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
    # results_plot_benchmark(["Original", "Singletons", "Triplets", "Biovec"],
    #                        ["Decision trees", "Random trees", "Nearest neighbours", "MLP", "Machine Learning"],
    #                        [[16.67, 11.1, 6.24, 0.34], [26.0, 22.44, 24.26, 5.97], [8.03, 3.29, 1.63, 0.22],
    #                         [199.47, 152.6, 52.42, 1.94], [67.14, 53.67, 50.93, 42.39]],
    #                        [[0.3306, 0.5661, 0.55, 0.668], [0.4, 0.6968, 0.6726, 0.8618],
    #                         [0.3935, 0.6532, 0.6161, 0.8302], [0.3806, 0.6565, 0.2677, 0.7917],
    #                         [0.4516128897666931, 0.5048387050628662, 0.2758064568042755, 0.5998011827468872]],
    #                        outfile="../presentation/images/benchmark.png")

    # plot_sizes(["../data/clean_dataset_original.pkl", "../data/clean_dataset_singletons.pkl",
    #             "../data/clean_dataset_triplets.pkl", "../data/clean_dataset_biovec.pkl"],
    #            ["Original", "Singletons", "Triplets", "Biovec"],
    #            "../presentation/images/sizes.png")
    #

    pass
