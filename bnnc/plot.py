import matplotlib.pyplot as pl

import numpy as np

COLOR_1 = "#11008f"
COLOR_1_LIGHT = "#8566bd"

COLOR_2 = "#ffa000"
COLOR_2_LIGHT = "#ffc57a"

def accuracy_by_samples(preds, labels):

    num_samples, num_predictions, num_classes = preds.shape
    r = []

    for ns in range(num_samples):
        p = preds[:ns + 1,:]
        cp = np.mean(p, axis=0).argmax(axis=1)
        correct = cp[cp == labels]
        r.append(correct.shape[0] / num_predictions)

    return np.arange(num_samples) + 1, r

def plot_accuracy_by_samples(plot_data, result_dir):
    fig, axes = pl.subplots(1,1)
    axes.plot(plot_data[0], plot_data[1], color=COLOR_1, zorder=3)
    axes.set_xlabel("Number of Samples")
    axes.set_ylabel("Accuracy")
    axes.grid(visible=True, axis='y', zorder=0)
    fig.tight_layout()
    fig.savefig(f'{result_dir}/acc_by_samples.pdf')
    fig.savefig(f'{result_dir}/acc_by_samples.png')
    return fig, axes

def plot_accuracy_by_samples_double(plots_data, result_dir):
    fig, axes = pl.subplots(1,1)
    plot_data = plots_data[1]
    axes.plot(plot_data[0], plot_data[1], color=COLOR_1, zorder=3, label="C")
    plot_data = plots_data[0]
    axes.plot(plot_data[0], plot_data[1], color=COLOR_2, zorder=3, label="Python")
    axes.set_xlabel("Number of Samples")
    axes.set_ylabel("Accuracy")
    axes.grid(visible=True, axis='y', zorder=0)
    axes.legend()
    fig.tight_layout()
    fig.savefig(f'{result_dir}/acc_by_samples.pdf')
    fig.savefig(f'{result_dir}/acc_by_samples.png')
    return fig, axes

def group_accuracy_data(metrics, w = 0.05):
    acc_data = []

    bins = np.arange(0, 1 + w, w)

    for i in range(len(bins) - 1):
        g = metrics[(metrics[:,4] >= bins[i]) & (metrics[:,4] < bins[i+1])]

        # Empty or too small group
        if g.shape[0] < 5:
            acc_data.append([float("nan"), float("nan")])

        else:  
            acc_data.append([np.sum(g[:,0]) / g.shape[0], g.shape[0] / metrics.shape[0]])

    # len(bins) - 1
    # bins, [bin accuracy, % of total data in bin]
    return (bins, w), np.array(acc_data)

def plot_accuracy_vs_uncertainty(plot_data, result_dir):
    fig, axes = pl.subplots(1,1)

    w = plot_data[0][1]

    # Plot
    axes.bar(plot_data[0][0][:-1], plot_data[1][:,1], width=w * 0.8, edgecolor='black', color=COLOR_1)
    axes.plot(plot_data[0][0][:-1], plot_data[1][:,0], color=COLOR_1)

    axes.set_ylim(0,1)

    axes.set_ylabel('% of pixels in group')
    secax_y = axes.secondary_yaxis(location="right")
    secax_y.set_ylabel('Group accuracy')
    axes.set_xlabel('Group Normalized Uncertainty')

    fig.tight_layout()
    fig.savefig(f'{result_dir}/accuracy_vs_uncertainty.pdf')
    fig.savefig(f'{result_dir}/accuracy_vs_uncertainty.png')
    return fig, axes

def plot_accuracy_vs_uncertainty_double(plots_data, result_dir):
    # Create fig
    fig, axes = pl.subplots(1,1)
    axes.set_ylim(0,1)
    
    w = plots_data[0][0][1]

    # Plot
    plot_data = plots_data[1]
    axes.bar(plot_data[0][0][:-1]-w/4, plot_data[1][:,1], width=w/2, edgecolor='black', color=COLOR_1)
    axes.plot(plot_data[0][0][:-1], plot_data[1][:,0], color=COLOR_1)

    # Repeat using same bins for python prediction
    plot_data = plots_data[0]
    axes.bar(plot_data[0][0][:-1]+w/4, plot_data[1][:,1], width=w/2, edgecolor='black', color=COLOR_2)
    axes.plot(plot_data[0][0][:-1], plot_data[1][:,0], color=COLOR_2)

    # Legend
    axes.legend(
        ['C', 'Python'],
        bbox_to_anchor=(0.5, 1),
        loc='lower center', ncols=2
    )

    axes.set_ylabel('% of pixels in group')

    secax_y = axes.secondary_yaxis(location="right")
    secax_y.set_ylabel('Group accuracy')

    axes.set_xlabel('Group Normalized Uncertainty')

    fig.tight_layout()
    fig.savefig(f'{result_dir}/accuracy_vs_uncertainty.pdf')
    fig.savefig(f'{result_dir}/accuracy_vs_uncertainty.png')
    return fig, axes



def class_uncertainty_data(metrics, num_classes, labels):

    vec_H = []
    vec_Ep = []

    for i in range(num_classes):
        mask = labels[:] == i
        vec_H.append(np.mean(metrics[mask,2]))
        vec_Ep.append(np.mean(metrics[mask,3]))
    
    x_bar = np.arange(num_classes) 

    return (x_bar, np.log(num_classes)), vec_H, vec_Ep

def plot_class_uncertainty(plot_data, result_dir):
    fig, axes = pl.subplots(1,1)
    w = 0.9
    axes.bar(plot_data[0][0], plot_data[1], width=w, color=COLOR_1_LIGHT, edgecolor='black', zorder=3)
    axes.bar(plot_data[0][0], plot_data[2], width=w, color=COLOR_1, edgecolor='black', zorder=3)

    axes.set_xticks(plot_data[0][0])
    axes.set_xlabel("Class")

    axes.set_ylim(0,plot_data[0][1])
    axes.set_ylabel('Average uncertainty')

    # Legend
    axes.legend(
        [r'$\mathbb{H}$', r'$\mathbb{E}_{p(w|D)}$'],
        bbox_to_anchor=(0.5, 1),
        loc='lower center', ncols=4
    )

    fig.tight_layout()
    fig.savefig(f'{result_dir}/class_uncertainty.pdf')
    fig.savefig(f'{result_dir}/class_uncertainty.png')
    return fig, axes

def plot_class_uncertainty_double(plots_data, result_dir):
    # Create fig
    fig, axes = pl.subplots(1,1)
    w = 0.4

    plot_data = plots_data[0]
    axes.bar(plot_data[0][0] - w/2, plot_data[1], width=w, color=COLOR_2_LIGHT, edgecolor='black', zorder=3)
    axes.bar(plot_data[0][0] - w/2, plot_data[2], width=w, color=COLOR_2, edgecolor='black', zorder=3)

    plot_data = plots_data[1]
    axes.bar(plot_data[0][0] + w/2, plot_data[1], width=w, color=COLOR_1_LIGHT, edgecolor='black', zorder=3)
    axes.bar(plot_data[0][0] + w/2, plot_data[2], width=w, color=COLOR_1, edgecolor='black', zorder=3)

    axes.set_xticks(plot_data[0][0])
    axes.set_xlabel("Class")

    axes.set_ylim(0,plot_data[0][1])
    axes.set_ylabel('Average uncertainty')
    
    # Legend
    axes.legend(
        [
            r'Python $\mathbb{H}$', r'Python $\mathbb{E}_{p(w|D)}$', 
            r'C $\mathbb{H}$', r'C $\mathbb{E}_{p(w|D)}$'
        ],
        bbox_to_anchor=(0.5, 1),
        loc='lower center', ncols=4
    )

    fig.tight_layout()
    fig.savefig(f'{result_dir}/class_uncertainty.pdf')
    fig.savefig(f'{result_dir}/class_uncertainty.png')
    return fig, axes


def calibration_data(averages, labels):
    _, num_classes =  averages.shape

    labels_one_hot = np.zeros((len(labels), num_classes))
    labels_one_hot[range(len(labels)), labels] = 1

    p_groups = np.arange(0, 1.1, 0.1)
    center = p_groups[:-1] + (p_groups[1:] - p_groups[:-1]) / 2

    result = []
    for i in range(len(p_groups) - 1):
        p_min = p_groups[i]
        p_max = p_groups[i + 1]
        group = labels_one_hot[(averages >= p_min) & (averages < p_max)]
        result.append(group.sum() / len(group))
    
    return (center, p_groups), result

def plot_calibration(plot_data, result_dir):
    fig, ax = pl.subplots(1,1)
    ax.plot(plot_data[0][0], plot_data[0][0], color='black', linestyle='dashed')
    ax.plot(plot_data[0][0], plot_data[1], color=COLOR_1)
    ax.legend(
        ["Optimal Calibration", 'Model'],
        bbox_to_anchor=(0.5, 1),
        loc='lower center', ncols=2
    )
    ax.grid(visible=True, axis='y', zorder=0)
    ax.set(xticks=plot_data[0][1], yticks=plot_data[0][1])

    fig.tight_layout()
    fig.savefig(f'{result_dir}/calibration.pdf')
    fig.savefig(f'{result_dir}/calibration.png')
    return fig, axes

def plot_calibration_double(plots_data, result_dir):

    fig, ax = pl.subplots(1,1)
    
    plot_data = plots_data[1]
    ax.plot(plot_data[0][0], plot_data[0][0], color='black', linestyle='dashed')
    ax.plot(plot_data[0][0], plot_data[1], color=COLOR_1)

    plot_data = plots_data[0]
    ax.plot(plot_data[0][0], plot_data[1], color=COLOR_2)

    ax.legend(
        ["Optimal Calibration", 'C', 'Python'],
        bbox_to_anchor=(0.5, 1),
        loc='lower center', ncols=3
    )

    ax.grid(visible=True, axis='y', zorder=0)
    ax.set(xticks=plot_data[0][1], yticks=plot_data[0][1])

    fig.tight_layout()
    fig.savefig(f'{result_dir}/calibration.pdf')
    fig.savefig(f'{result_dir}/calibration.png')
    return fig, ax



def prediction_uncertainty_data(metrics, w = 0.01):
    mh = np.max(metrics[:,2])
    bins = np.arange(0, mh + w, w)
    data_correct = metrics[metrics[:,0] == 1.0]
    data_fail = metrics[metrics[:,0] == 0.0]
    count_correct, _ = np.histogram(data_fail[:,2], bins=bins)
    count_fail, _ = np.histogram(data_correct[:,2], bins=bins)
    return bins, count_correct, count_fail

def plot_prediction_uncertainty(plot_data, result_dir):
    fig, axes = pl.subplots(1,1)
    axes.plot(plot_data[0][:-1], plot_data[1], color='red', zorder=3, label="Incorrect predictions")
    axes.plot(plot_data[0][:-1], plot_data[2], color='royalblue', zorder=3, label="Correct predictions")
    axes.grid(visible=True, axis='y', zorder=0)
    axes.legend()
    fig.supxlabel('Prediction Uncertainty')
    fig.supylabel('Density')
    fig.tight_layout()
    fig.savefig(f'{result_dir}/prediction_uncertainty.pdf')
    fig.savefig(f'{result_dir}/prediction_uncertainty.png')
    return fig, axes

def plot_prediction_uncertainty_double(plots_data, result_dir):
    fig, axes = pl.subplots(1,2)

    plot_data = plots_data[0]
    ax = axes[0]
    ax.plot(plot_data[0][:-1], plot_data[1], color='red', zorder=3)
    ax.plot(plot_data[0][:-1], plot_data[2], color='royalblue', zorder=3)

    ax.set_title("Python")
    ax.grid(visible=True, axis='y', zorder=0)

    plot_data = plots_data[1]
    ax = axes[1]
    ax.plot(plot_data[0][:-1], plot_data[1], color='red', zorder=3, label="Incorrect predictions")
    ax.plot(plot_data[0][:-1], plot_data[2], color='royalblue', zorder=3, label="Correct predictions")

    ax.set_title("C")
    ax.grid(visible=True, axis='y', zorder=0)
    ax.legend()

    fig.supxlabel('Prediction Uncertainty')
    fig.supylabel('Density')

    fig.tight_layout()
    fig.savefig(f'{result_dir}/prediction_uncertainty.pdf')
    fig.savefig(f'{result_dir}/prediction_uncertainty.png')
    return fig, axes


def accuracy_vs_uncertainty_data(metrics, nths = 50):

    nsamples = metrics.shape[0]

    metrics_correct = metrics[metrics[:,0] == 1]
    metrics_error = metrics[metrics[:,0] == 0]

    step_size = 1 / nths
    uths = np.arange(0, 1 + step_size, step_size)

    r = np.zeros((nths, 2))

    for i, uth in enumerate(uths):
        nau = metrics_correct[metrics_correct[:,4] > uth].shape[0]
        nac = metrics_correct[metrics_correct[:,4] <= uth].shape[0]
        nic = metrics_error[metrics_error[:,4] > uth].shape[0]
        niu = metrics_error[metrics_error[:,4] <= uth].shape[0]

        p_acc_cert = nac / (nac + nic)
        p_unc_inn = niu / (nic + niu)

        r[i, 0] = p_acc_cert
        r[i, 1] = p_unc_inn

    return uths, r

def free_plot_memory():
    pl.close("all")


def compare_predictions_plots(data_a, data_b, labels, result_dir):
    metrics_a, averages_a, preds_a = data_a
    metrics_b, averages_b, preds_b = data_b
    _, num_classes = averages_a.shape
    plot_data = (
        prediction_uncertainty_data(metrics_a),
        prediction_uncertainty_data(metrics_b)
    )
    f1 = plot_prediction_uncertainty_double(plot_data, result_dir)
    plot_data = (
        calibration_data(averages_a, labels),
        calibration_data(averages_b, labels)
    )
    f2 = plot_calibration_double(plot_data, result_dir)
    plot_data = (
        group_accuracy_data(metrics_a),
        group_accuracy_data(metrics_b)
    )
    f3 = plot_accuracy_vs_uncertainty_double(plot_data, result_dir)
    plot_data = (
        class_uncertainty_data(metrics_a, num_classes, labels),
        class_uncertainty_data(metrics_b, num_classes, labels)
    )
    f4 = plot_class_uncertainty_double(plot_data, result_dir)
    plot_data = (
        accuracy_by_samples(preds_a, labels),
        accuracy_by_samples(preds_b, labels)
    )
    f5 = plot_accuracy_by_samples_double(plot_data, result_dir)
    return f1, f2, f3, f4, f5


def all_plots(data, labels, result_dir):
    metrics, averages, predictions = data
    _, num_classes = averages.shape
    f1 = plot_prediction_uncertainty(prediction_uncertainty_data(metrics), result_dir)
    f2 = plot_calibration(calibration_data(averages, labels), result_dir)
    f3 = plot_class_uncertainty(class_uncertainty_data(metrics, num_classes, labels), result_dir)
    f4 = plot_accuracy_vs_uncertainty(group_accuracy_data(metrics), result_dir)
    f5 = plot_accuracy_by_samples(accuracy_by_samples(predictions, labels), result_dir)
    return f1, f2, f3, f4, f5
