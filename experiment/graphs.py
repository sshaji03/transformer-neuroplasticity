import os
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = "results"


def save_fig(filename):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    plt.savefig(os.path.join(RESULTS_DIR, filename))
    plt.close()


def plot_training_curves(neuro_results, normal_results, task_names):
    fig, (ax_loss, ax_acc) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    for ax, metric, title in [
        (ax_loss, "loss", "Training Loss Across Tasks"),
        (ax_acc, "accuracy", "Training Accuracy Across Tasks"),
    ]:
        boundaries = [0]
        for results, label, color in [
            (neuro_results, "Neuroplasticity", "blue"),
            (normal_results, "Normal", "orange"),
        ]:
            all_values = []
            boundaries = [0]
            for task_name in task_names:
                values = [epoch[metric] for epoch in results[task_name]["training_curves"]]
                all_values.extend(values)
                boundaries.append(len(all_values))
            ax.plot(all_values, label=label, color=color)

        for b in boundaries[1:-1]:
            ax.axvline(x=b, color="gray", linestyle="--", linewidth=0.8)
        for i, task_name in enumerate(task_names):
            mid = (boundaries[i] + boundaries[i + 1]) / 2
            ax.text(mid, 1.02, task_name.split("/")[-1], ha="center", va="bottom",
                    fontsize=7, transform=ax.get_xaxis_transform())

        ax.set_title(title)
        ax.set_ylabel(metric.capitalize())
        ax.legend()

    ax_acc.set_xlabel("Epoch")
    plt.tight_layout()
    save_fig("training_curves.png")


def plot_post_training_accuracy(neuro_results, normal_results, task_names):
    fig, axes = plt.subplots(1, len(task_names), figsize=(4 * len(task_names), 5), sharey=True, squeeze=False)
    axes = axes[0]

    for ax, eval_task in zip(axes, task_names):
        neuro_accs = []
        normal_accs = []
        x_labels = []

        for task_name in task_names:
            post_neuro = neuro_results[task_name]["post_training"]
            post_normal = normal_results[task_name]["post_training"]
            if eval_task in post_neuro:
                neuro_accs.append(post_neuro[eval_task]["accuracy"])
                normal_accs.append(post_normal[eval_task]["accuracy"])
                x_labels.append(task_name.split("/")[-1])

        x = np.arange(len(x_labels))
        ax.bar(x - 0.2, neuro_accs, width=0.4, label="Neuroplasticity", color="blue", alpha=0.7)
        ax.bar(x + 0.2, normal_accs, width=0.4, label="Normal", color="orange", alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=30, ha="right", fontsize=8)
        ax.set_title(f"Acc on {eval_task.split('/')[-1]}")
        ax.set_ylim(0, 1)

    axes[0].set_ylabel("Accuracy")
    axes[-1].legend()
    plt.suptitle("Post-Training Accuracy on Each Task")
    plt.tight_layout()
    save_fig("post_training_accuracy.png")


def plot_checkpoint_accuracy(neuro_results, task_names):
    checkpoints = ["post_training", "checkpoint_1", "checkpoint_2", "checkpoint_3", "checkpoint_4"]
    colors = ["black", "blue", "green", "red", "purple"]

    fig, axes = plt.subplots(1, len(task_names), figsize=(4 * len(task_names), 5), sharey=True, squeeze=False)
    axes = axes[0]

    for ax, eval_task in zip(axes, task_names):
        for cp, color in zip(checkpoints, colors):
            accs = []
            x_labels = []
            for task_name in task_names:
                if cp in neuro_results[task_name] and eval_task in neuro_results[task_name][cp]:
                    accs.append(neuro_results[task_name][cp][eval_task]["accuracy"])
                    x_labels.append(task_name.split("/")[-1])
            if accs:
                ax.plot(x_labels, accs, marker="o", label=cp.replace("_", " "), color=color)

        ax.set_title(f"Acc on {eval_task.split('/')[-1]}")
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=30)

    axes[0].set_ylabel("Accuracy")
    axes[-1].legend(fontsize=7)
    plt.suptitle("Checkpoint Accuracy per Task Over Training")
    plt.tight_layout()
    save_fig("checkpoint_accuracy.png")


def plot_average_accuracy(neuro_results, normal_results, task_names):
    checkpoints = ["checkpoint_1", "checkpoint_2", "checkpoint_3", "checkpoint_4"]
    colors = ["blue", "green", "red", "purple"]

    fig, ax = plt.subplots(figsize=(10, 5))

    neuro_avg = []
    normal_avg = []
    x_labels = []

    for task_name in task_names:
        post_neuro = neuro_results[task_name]["post_training"]
        post_normal = normal_results[task_name]["post_training"]
        neuro_avg.append(np.mean([v["accuracy"] for v in post_neuro.values()]))
        normal_avg.append(np.mean([v["accuracy"] for v in post_normal.values()]))
        x_labels.append(task_name.split("/")[-1])

    ax.plot(x_labels, normal_avg, marker="o", label="Normal", color="orange")
    ax.plot(x_labels, neuro_avg, marker="o", label="Post-training (Neuro)", color="black")

    for cp, color in zip(checkpoints, colors):
        cp_avg = []
        cp_labels = []
        for task_name in task_names:
            if cp in neuro_results[task_name]:
                vals = neuro_results[task_name][cp].values()
                cp_avg.append(np.mean([v["accuracy"] for v in vals]))
                cp_labels.append(task_name.split("/")[-1])
        if cp_avg:
            ax.plot(cp_labels, cp_avg, marker="o", label=cp.replace("_", " "), color=color)

    ax.set_ylabel("Average Accuracy")
    ax.set_xlabel("After Task")
    ax.set_ylim(0, 1)
    ax.set_title("Average Accuracy Across All Seen Tasks")
    ax.legend()
    plt.tight_layout()
    save_fig("average_accuracy.png")


def plot_accuracy_heatmap(neuro_results, task_names, checkpoint="post_training"):
    n = len(task_names)
    matrix = np.full((n, n), np.nan)

    for col_idx, task_trained in enumerate(task_names):
        data = neuro_results[task_trained].get(checkpoint, {})
        for row_idx, task_eval in enumerate(task_names):
            if task_eval in data:
                matrix[row_idx, col_idx] = data[task_eval]["accuracy"]

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, vmin=0, vmax=1, cmap="Blues")
    plt.colorbar(im, ax=ax, label="Accuracy")

    short_names = [t.split("/")[-1] for t in task_names]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(short_names, rotation=30, ha="right")
    ax.set_yticklabels(short_names)
    ax.set_xlabel("After Training Task")
    ax.set_ylabel("Evaluated On Task")
    ax.set_title(f"Accuracy Heatmap — {checkpoint.replace('_', ' ').title()}")

    for i in range(n):
        for j in range(n):
            if not np.isnan(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=8)

    plt.tight_layout()
    save_fig(f"heatmap_{checkpoint}.png")


def plot_all(neuro_results, normal_results, task_names):
    plot_training_curves(neuro_results, normal_results, task_names)
    plot_post_training_accuracy(neuro_results, normal_results, task_names)
    plot_checkpoint_accuracy(neuro_results, task_names)
    plot_average_accuracy(neuro_results, normal_results, task_names)
    for cp in ["post_training", "checkpoint_1", "checkpoint_2", "checkpoint_3", "checkpoint_4"]:
        plot_accuracy_heatmap(neuro_results, task_names, checkpoint=cp)
