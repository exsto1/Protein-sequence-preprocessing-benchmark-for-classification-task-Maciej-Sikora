from matplotlib import pyplot as plt


def input_analysis(input_file="../data/clustering/data_file_clustered.fasta", outfile="../presentation/images/histogram.png"):
    with open(input_file) as fileh:
        file = fileh.readlines()

    file = [i.rstrip() for i in file if i]
    names = [i for i in file if ">" in i]
    seq = [i for i in file if ">" not in i]

    family_data = {}
    for i in names:
        fam = i.split("_")[1]
        if fam in family_data:
            family_data[fam] += 1
        else:
            family_data[fam] = 1

    fam_names = list(family_data.keys())
    fam_names.sort(key=lambda i: family_data[i], reverse=True)
    hist_data = {}
    for i in fam_names:
        hist_val = family_data[i] - (family_data[i] % 100)
        if hist_val > 1000:
            if 1000 in hist_data:
                hist_data[1000] += 1
            else:
                hist_data[1000] = 1
        else:
            if hist_val in hist_data:
                hist_data[hist_val] += 1
            else:
                hist_data[hist_val] = 1

    x_ticks = list(hist_data.keys())
    x_ticks.sort()
    x_ticks_labels = [f"{i}-{i+99}" for i in x_ticks]
    x_ticks_labels[-1] = "1000+"

    values = [hist_data[i] for i in x_ticks]

    hatches = ["/" for i in range(len(x_ticks))]
    hatches[-1] = "////"


    fig, ax = plt.subplots(figsize=(7,4))
    plt.bar(x_ticks, values, width=100,
            edgecolor="k", facecolor="w", hatch=hatches)
    plt.xticks(x_ticks, x_ticks_labels, rotation=-45, ha="left")
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    plt.xlabel("Number of sequences in the family")
    plt.xlabel("Number of cases (families)")
    plt.title("Histogram of family sizes in the input data")
    plt.tight_layout()
    plt.savefig(outfile)
    plt.show()


if __name__ == '__main__':
    input_analysis()
