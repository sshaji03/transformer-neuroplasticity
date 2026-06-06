import os
import matplotlib.pyplot as plt
import numpy as np

DEFAULT_RESULTS_DIR = "results"


def save_fig(filename, results_dir=DEFAULT_RESULTS_DIR):
    """Saves the file to the results directory"""
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(os.path.join(results_dir, filename))
    plt.close()


def plot_training_curves(neuro_results, normal_results, task_names, results_dir=DEFAULT_RESULTS_DIR,
                         memory_results=None):
    """Plots loss and accuracy training curves across all tasks for each model, dividing graph by task"""
    fig, (ax_loss, ax_acc) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    boundaries = [0]
    for task_name in task_names:
        n_epochs = len(neuro_results[task_name]["training_curves"])
        boundaries.append(boundaries[-1] + n_epochs)

    for ax, metric, title in [
        (ax_loss, "loss", "Training Loss Across Tasks"),
        (ax_acc, "accuracy", "Training Accuracy Across Tasks"),
    ]:
        series = [
            (neuro_results, "Neuroplasticity", "blue"),
            (normal_results, "Normal", "orange"),
        ]
        if memory_results is not None:
            series.append((memory_results, "Memory Replay", "teal"))
        for results, label, color in series:
            all_values = []
            for task_name in task_names:
                all_values.extend(epoch[metric] for epoch in results[task_name]["training_curves"])
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
    save_fig("training_curves.png", results_dir)


def plot_post_training_accuracy(neuro_results, normal_results, task_names, results_dir=DEFAULT_RESULTS_DIR):
    """Bar charts comparing neuroplasticity vs normal accuracy after training each task for each seen task"""
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
    save_fig("post_training_accuracy.png", results_dir)


def plot_checkpoint_accuracy(neuro_results, normal_results, task_names, results_dir=DEFAULT_RESULTS_DIR,
                             memory_results=None):
    """Plots accuracy on test data across tasks for each baseline and checkpoints"""
    checkpoints = ["post_training", "checkpoint_1", "checkpoint_2", "checkpoint_3", "checkpoint_4"]
    cp_colors = ["black", "blue", "green", "red", "purple"]

    fig, axes = plt.subplots(1, len(task_names), figsize=(4 * len(task_names), 5), sharey=True, squeeze=False)
    axes = axes[0]

    for ax, eval_task in zip(axes, task_names):
        normal_accs, normal_x = [], []
        for task_name in task_names:
            post = normal_results[task_name]["post_training"]
            if eval_task in post:
                normal_accs.append(post[eval_task]["accuracy"])
                normal_x.append(task_name.split("/")[-1])
        if normal_accs:
            ax.plot(normal_x, normal_accs, marker="s", linestyle="--",
                    label="Normal (baseline)", color="orange", linewidth=1.5)

        if memory_results is not None:
            mem_accs, mem_x = [], []
            for task_name in task_names:
                post = memory_results[task_name]["post_training"]
                if eval_task in post:
                    mem_accs.append(post[eval_task]["accuracy"])
                    mem_x.append(task_name.split("/")[-1])
            if mem_accs:
                ax.plot(mem_x, mem_accs, marker="D", linestyle="-.",
                        label="Memory Replay", color="teal", linewidth=1.5)

        for cp, color in zip(checkpoints, cp_colors):
            accs, x_labels = [], []
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
    plt.suptitle("Checkpoint Accuracy vs Normal Baseline per Task Over Training")
    plt.tight_layout()
    save_fig("checkpoint_accuracy.png", results_dir)


def plot_average_accuracy(neuro_results, normal_results, task_names, results_dir=DEFAULT_RESULTS_DIR,
                          memory_results=None, equal_compute_results=None):
    """Plots average accuracy across all tasks after training each task for all baselines and checkpoints except 1"""
    checkpoints = ["checkpoint_1", "checkpoint_2", "checkpoint_3", "checkpoint_4"]
    colors = ["blue", "green", "red", "purple"]

    fig, ax = plt.subplots(figsize=(10, 5))

    neuro_avg, normal_avg, x_labels = [], [], []
    for task_name in task_names:
        post_neuro = neuro_results[task_name]["post_training"]
        post_normal = normal_results[task_name]["post_training"]
        neuro_avg.append(np.mean([v["accuracy"] for v in post_neuro.values()]))
        normal_avg.append(np.mean([v["accuracy"] for v in post_normal.values()]))
        x_labels.append(task_name.split("/")[-1])

    ax.plot(x_labels, normal_avg, marker="s", linestyle="--", label="Normal (baseline)", color="orange", linewidth=1.5)
    ax.plot(x_labels, neuro_avg, marker="o", label="Post-training (Neuro)", color="black")

    all_vals = list(neuro_avg) + list(normal_avg)

    if memory_results is not None:
        mem_avg = [
            np.mean([v["accuracy"] for v in memory_results[t]["post_training"].values()])
            for t in task_names
        ]
        ax.plot(x_labels, mem_avg, marker="D", linestyle="-.", label="Memory Replay", color="teal", linewidth=1.5)
        all_vals.extend(mem_avg)

    if equal_compute_results is not None:
        eq_avg = [
            np.mean([v["accuracy"] for v in equal_compute_results[t]["post_training"].values()])
            for t in task_names
        ]
        ax.plot(x_labels, eq_avg, marker="^", linestyle="-.", label="Equal-Compute Replay", color="chocolate", linewidth=1.5)
        all_vals.extend(eq_avg)
    for cp, color in zip(checkpoints, colors):
        cp_avg, cp_labels = [], []
        for task_name in task_names:
            if cp in neuro_results[task_name]:
                vals = neuro_results[task_name][cp].values()
                cp_avg.append(np.mean([v["accuracy"] for v in vals]))
                cp_labels.append(task_name.split("/")[-1])
        if cp_avg:
            ax.plot(cp_labels, cp_avg, marker="o", label=cp.replace("_", " "), color=color)
            all_vals.extend(cp_avg)

    floor = max(0.0, min(all_vals) - 0.05)
    ax.set_ylim(floor, 1.0)
    ax.set_ylabel("Average Accuracy")
    ax.set_xlabel("After Task")
    ax.set_title("Average Accuracy Across All Seen Tasks")
    ax.legend()
    plt.tight_layout()
    save_fig("average_accuracy.png", results_dir)


def plot_forgetting(neuro_results, normal_results, task_names, results_dir=DEFAULT_RESULTS_DIR,
                    memory_results=None, equal_compute_results=None):
    """Plots forgetting (difference in accuracy after initial training) for all baselines and checkpoints as new tasks
    are trained"""
    eval_tasks = task_names[:-1]
    fig, axes = plt.subplots(1, len(eval_tasks), figsize=(4 * len(eval_tasks), 5), sharey=True, squeeze=False)
    axes = axes[0]

    for ax, eval_task in zip(axes, eval_tasks):
        eval_idx = task_names.index(eval_task)
        subsequent = task_names[eval_idx + 1:]
        x_labels = [t.split("/")[-1] for t in subsequent]

        normal_base = normal_results[eval_task]["post_training"][eval_task]["accuracy"]
        neuro_base = neuro_results[eval_task]["post_training"][eval_task]["accuracy"]

        def drop(results, key, base):
            return [
                base - results[t].get(key, {}).get(eval_task, {}).get("accuracy", base)
                for t in subsequent
            ]

        ax.plot(x_labels, drop(normal_results, "post_training", normal_base),
                marker="s", linestyle="--", label="Normal", color="orange", linewidth=1.5)
        if memory_results is not None:
            mem_base = memory_results[eval_task]["post_training"][eval_task]["accuracy"]
            ax.plot(x_labels, drop(memory_results, "post_training", mem_base),
                    marker="D", linestyle="-.", label="Memory Replay", color="teal", linewidth=1.5)
        if equal_compute_results is not None:
            eq_base = equal_compute_results[eval_task]["post_training"][eval_task]["accuracy"]
            ax.plot(x_labels, drop(equal_compute_results, "post_training", eq_base),
                    marker="^", linestyle="-.", label="Equal-Compute Replay", color="chocolate", linewidth=1.5)
        ax.plot(x_labels, drop(neuro_results, "post_training", neuro_base),
                marker="o", label="Neuro (post-train)", color="black")
        ax.plot(x_labels, drop(neuro_results, "checkpoint_2", neuro_base),
                marker="o", label="CP2", color="green")
        ax.plot(x_labels, drop(neuro_results, "checkpoint_3", neuro_base),
                marker="o", label="CP3", color="red")
        ax.plot(x_labels, drop(neuro_results, "checkpoint_4", neuro_base),
                marker="o", label="CP4", color="purple")

        ax.axhline(0, color="gray", linestyle=":", linewidth=0.8)
        ax.set_title(f"Forgetting on\n{eval_task.split('/')[-1]}")
        ax.tick_params(axis="x", rotation=30)

    axes[0].set_ylabel("Accuracy Drop (higher = more forgetting)")
    axes[-1].legend(fontsize=7)
    plt.suptitle("Catastrophic Forgetting: Accuracy Drop on Previous Tasks")
    plt.tight_layout()
    save_fig("forgetting.png", results_dir)


def plot_delta_heatmap(neuro_results, normal_results, task_names, results_dir=DEFAULT_RESULTS_DIR):
    """Plots a heatmap of accuracy difference between CP3 and the normal baseline, separated by which task was trained"""
    n = len(task_names)
    matrix = np.full((n, n), np.nan)

    for col_idx, task_trained in enumerate(task_names):
        cp3_data = neuro_results[task_trained].get("checkpoint_3", {})
        normal_data = normal_results[task_trained].get("post_training", {})
        for row_idx, task_eval in enumerate(task_names):
            if task_eval in cp3_data and task_eval in normal_data:
                matrix[row_idx, col_idx] = (
                    cp3_data[task_eval]["accuracy"] - normal_data[task_eval]["accuracy"]
                )

    vabs = max(0.01, np.nanmax(np.abs(matrix)))
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, vmin=-vabs, vmax=vabs, cmap="RdYlGn")
    plt.colorbar(im, ax=ax, label="Accuracy Δ (CP3 − Normal)")

    short_names = [t.split("/")[-1] for t in task_names]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(short_names, rotation=30, ha="right")
    ax.set_yticklabels(short_names)
    ax.set_xlabel("After Training Task")
    ax.set_ylabel("Evaluated On Task")
    ax.set_title("Accuracy Gain: CP3 vs Normal Baseline (green = neuro wins)")

    for i in range(n):
        for j in range(n):
            if not np.isnan(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j]:+.2f}", ha="center", va="center", fontsize=8)

    plt.tight_layout()
    save_fig("heatmap_cp3_vs_normal.png", results_dir)


def plot_forgetting_heatmap(neuro_results, normal_results, task_names, checkpoint="checkpoint_3",
                            results_dir=DEFAULT_RESULTS_DIR):
    """Plots a heatmap of forgetting difference between CP3 and the normal baseline, separated by which task was trained."""
    n = len(task_names)
    matrix = np.full((n, n), np.nan)

    for col_idx, after_task in enumerate(task_names):
        neuro_cp = neuro_results[after_task].get(checkpoint, {})
        normal_post = normal_results[after_task]["post_training"]
        for row_idx, eval_task in enumerate(task_names):
            if row_idx >= col_idx:  # only past tasks can be forgotten
                continue
            neuro_base   = neuro_results[eval_task]["post_training"][eval_task]["accuracy"]
            normal_base  = normal_results[eval_task]["post_training"][eval_task]["accuracy"]
            neuro_after  = neuro_cp.get(eval_task, {}).get("accuracy")
            normal_after = normal_post.get(eval_task, {}).get("accuracy")
            if neuro_after is not None and normal_after is not None:
                normal_forgetting = normal_base - normal_after
                neuro_forgetting  = neuro_base  - neuro_after
                matrix[row_idx, col_idx] = normal_forgetting - neuro_forgetting

    vabs = max(0.01, np.nanmax(np.abs(matrix)))
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, vmin=-vabs, vmax=vabs, cmap="RdYlGn")
    plt.colorbar(im, ax=ax, label="Forgetting Δ (Normal − Neuro)  [green = neuro retains better]")

    short_names = [t.split("/")[-1] for t in task_names]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(short_names, rotation=30, ha="right")
    ax.set_yticklabels(short_names)
    ax.set_xlabel("After Training Task")
    ax.set_ylabel("Evaluated On Task")
    ax.set_title(f"Forgetting Reduction: Normal − {checkpoint.replace('_', ' ').title()}")

    for i in range(n):
        for j in range(n):
            if not np.isnan(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j]:+.2f}", ha="center", va="center", fontsize=8)

    plt.tight_layout()
    save_fig(f"heatmap_forgetting_{checkpoint}.png", results_dir)


def plot_accuracy_heatmap(neuro_results, task_names, checkpoint="post_training", results_dir=DEFAULT_RESULTS_DIR):
    """Plots a heatmap of accuracy for each model version, separated by which task was trained"""
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
    save_fig(f"heatmap_{checkpoint}.png", results_dir)


def plot_k_comparison(all_k_results, normal_results, task_names, results_dir=DEFAULT_RESULTS_DIR,
                      memory_results=None, equal_compute_results=None):
    """Plots average accuracy and average forgetting at the end of continual learning across different top-k
    values."""
    k_values = sorted(all_k_results.keys())
    last_task = task_names[-1]

    def avg_acc_final(neuro_results, checkpoint):
        data = neuro_results[last_task].get(checkpoint, {})
        vals = [data[t]["accuracy"] for t in task_names if t in data]
        return np.mean(vals) if vals else np.nan

    def avg_forgetting(neuro_results, checkpoint):
        drops = []
        for eval_task in task_names[:-1]:
            base = neuro_results[eval_task]["post_training"][eval_task]["accuracy"]
            final = neuro_results[last_task].get(checkpoint, {}).get(eval_task, {}).get("accuracy")
            if final is not None:
                drops.append(base - final)
        return np.mean(drops) if drops else np.nan

    normal_final = np.mean([
        normal_results[last_task]["post_training"][t]["accuracy"]
        for t in task_names if t in normal_results[last_task]["post_training"]
    ])
    normal_forget = np.mean([
        normal_results[t]["post_training"][t]["accuracy"]
        - normal_results[last_task]["post_training"].get(t, {}).get("accuracy",
          normal_results[t]["post_training"][t]["accuracy"])
        for t in task_names[:-1]
        if t in normal_results[last_task]["post_training"]
    ])

    fig, (ax_acc, ax_forget) = plt.subplots(1, 2, figsize=(12, 5))

    for cp, color, label in [
        ("post_training", "black", "Neuro post-train"),
        ("checkpoint_2", "green", "CP2"),
        ("checkpoint_3", "red", "CP3"),
        ("checkpoint_4", "purple", "CP4"),
    ]:
        accs = [avg_acc_final(all_k_results[k], cp) for k in k_values]
        ax_acc.plot(k_values, accs, marker="o", color=color, label=label)

    ax_acc.axhline(normal_final, linestyle="--", color="orange", linewidth=1.5,
                   label=f"Normal baseline ({normal_final:.2f})")
    if memory_results is not None:
        mem_final = np.mean([
            memory_results[last_task]["post_training"][t]["accuracy"]
            for t in task_names if t in memory_results[last_task]["post_training"]
        ])
        ax_acc.axhline(mem_final, linestyle="-.", color="teal", linewidth=1.5,
                       label=f"Memory Replay ({mem_final:.2f})")
    if equal_compute_results is not None:
        eq_final = np.mean([
            equal_compute_results[last_task]["post_training"][t]["accuracy"]
            for t in task_names if t in equal_compute_results[last_task]["post_training"]
        ])
        ax_acc.axhline(eq_final, linestyle="-.", color="chocolate", linewidth=1.5,
                       label=f"Equal-Compute Replay ({eq_final:.2f})")
    ax_acc.set_xlabel("top-k fraction (proportion of weights kept as important)")
    ax_acc.set_ylabel("Average Accuracy across all tasks")
    ax_acc.set_title("Final Average Accuracy vs k")
    ax_acc.set_xticks(k_values)
    ax_acc.legend(fontsize=8)

    for cp, color, label in [
        ("post_training", "black", "Neuro post-train"),
        ("checkpoint_3", "red", "CP3"),
        ("checkpoint_4", "purple", "CP4"),
    ]:
        forgetting = [avg_forgetting(all_k_results[k], cp) for k in k_values]
        ax_forget.plot(k_values, forgetting, marker="o", color=color, label=label)

    ax_forget.axhline(normal_forget, linestyle="--", color="orange", linewidth=1.5,
                      label=f"Normal baseline ({normal_forget:.2f})")
    if memory_results is not None:
        mem_forget = np.mean([
            memory_results[t]["post_training"][t]["accuracy"]
            - memory_results[last_task]["post_training"].get(t, {}).get("accuracy",
              memory_results[t]["post_training"][t]["accuracy"])
            for t in task_names[:-1]
            if t in memory_results[last_task]["post_training"]
        ])
        ax_forget.axhline(mem_forget, linestyle="-.", color="teal", linewidth=1.5,
                          label=f"Memory Replay ({mem_forget:.2f})")
    if equal_compute_results is not None:
        eq_forget = np.mean([
            equal_compute_results[t]["post_training"][t]["accuracy"]
            - equal_compute_results[last_task]["post_training"].get(t, {}).get("accuracy",
              equal_compute_results[t]["post_training"][t]["accuracy"])
            for t in task_names[:-1]
            if t in equal_compute_results[last_task]["post_training"]
        ])
        ax_forget.axhline(eq_forget, linestyle="-.", color="chocolate", linewidth=1.5,
                          label=f"Equal-Compute Replay ({eq_forget:.2f})")
    ax_forget.set_xlabel("top-k fraction (proportion of weights kept as important)")
    ax_forget.set_ylabel("Average Forgetting (accuracy drop on previous tasks)")
    ax_forget.set_title("Average Forgetting vs k")
    ax_forget.set_xticks(k_values)
    ax_forget.legend(fontsize=8)

    plt.suptitle("Neuroplasticity Performance across k values")
    plt.tight_layout()
    save_fig("k_comparison.png", results_dir)


def plot_epochs_comparison(all_neuro_results, all_normal_results, task_names, k_values,
                           results_dir=DEFAULT_RESULTS_DIR, all_memory_results=None,
                           all_equal_compute_results=None):
    """Plots average accuracy and forgetting between 3, 5, 10 epochs"""
    epoch_values = sorted(all_neuro_results.keys())
    last_task = task_names[-1]

    def avg_acc_final(neuro_results, checkpoint):
        data = neuro_results[last_task].get(checkpoint, {})
        vals = [data[t]["accuracy"] for t in task_names if t in data]
        return np.mean(vals) if vals else np.nan

    def avg_forgetting(neuro_results, checkpoint):
        drops = []
        for eval_task in task_names[:-1]:
            base = neuro_results[eval_task]["post_training"][eval_task]["accuracy"]
            final = neuro_results[last_task].get(checkpoint, {}).get(eval_task, {}).get("accuracy")
            if final is not None:
                drops.append(base - final)
        return np.mean(drops) if drops else np.nan

    def normal_avg_acc(normal_results):
        data = normal_results[last_task]["post_training"]
        return np.mean([data[t]["accuracy"] for t in task_names if t in data])

    def normal_avg_forgetting(normal_results):
        drops = []
        for eval_task in task_names[:-1]:
            base = normal_results[eval_task]["post_training"][eval_task]["accuracy"]
            final = normal_results[last_task]["post_training"].get(eval_task, {}).get("accuracy")
            if final is not None:
                drops.append(base - final)
        return np.mean(drops) if drops else np.nan

    # One column per k value so the plots stay readable
    n_k = len(k_values)
    fig, axes = plt.subplots(2, n_k, figsize=(6 * n_k, 9), squeeze=False)

    cp_styles = [
        ("post_training", "black", "Neuro post-train"),
        ("checkpoint_2",  "green",  "CP2"),
        ("checkpoint_3",  "red",    "CP3"),
        ("checkpoint_4",  "purple", "CP4"),
    ]

    for col, k in enumerate(k_values):
        ax_acc    = axes[0][col]
        ax_forget = axes[1][col]

        normal_accs       = [normal_avg_acc(all_normal_results[e])       for e in epoch_values]
        normal_forgetting = [normal_avg_forgetting(all_normal_results[e]) for e in epoch_values]

        ax_acc.plot(epoch_values, normal_accs, marker="s", linestyle="--",
                    color="orange", linewidth=1.5, label="Normal (baseline)")
        ax_forget.plot(epoch_values, normal_forgetting, marker="s", linestyle="--",
                       color="orange", linewidth=1.5, label="Normal (baseline)")

        if all_memory_results is not None:
            mem_accs       = [normal_avg_acc(all_memory_results[e])       for e in epoch_values]
            mem_forgetting = [normal_avg_forgetting(all_memory_results[e]) for e in epoch_values]
            ax_acc.plot(epoch_values, mem_accs, marker="D", linestyle="-.",
                        color="teal", linewidth=1.5, label="Memory Replay")
            ax_forget.plot(epoch_values, mem_forgetting, marker="D", linestyle="-.",
                           color="teal", linewidth=1.5, label="Memory Replay")

        if all_equal_compute_results is not None:
            eq_accs       = [normal_avg_acc(all_equal_compute_results[e])       for e in epoch_values]
            eq_forgetting = [normal_avg_forgetting(all_equal_compute_results[e]) for e in epoch_values]
            ax_acc.plot(epoch_values, eq_accs, marker="^", linestyle="-.",
                        color="chocolate", linewidth=1.5, label="Equal-Compute Replay")
            ax_forget.plot(epoch_values, eq_forgetting, marker="^", linestyle="-.",
                           color="chocolate", linewidth=1.5, label="Equal-Compute Replay")

        for cp, color, label in cp_styles:
            accs       = [avg_acc_final(all_neuro_results[e][k], cp)  for e in epoch_values]
            forgetting = [avg_forgetting(all_neuro_results[e][k], cp) for e in epoch_values]
            ax_acc.plot(epoch_values, accs, marker="o", color=color, label=label)
            ax_forget.plot(epoch_values, forgetting, marker="o", color=color, label=label)

        ax_acc.set_title(f"Final Avg Accuracy  (k={k})")
        ax_acc.set_ylabel("Average Accuracy")
        ax_acc.set_xticks(epoch_values)
        ax_acc.legend(fontsize=7)

        ax_forget.set_title(f"Average Forgetting  (k={k})")
        ax_forget.set_ylabel("Accuracy Drop on Previous Tasks")
        ax_forget.set_xlabel("Epochs per task")
        ax_forget.set_xticks(epoch_values)
        ax_forget.legend(fontsize=7)

    plt.suptitle("Effect of Training Epochs on Neuroplasticity Performance")
    plt.tight_layout()
    save_fig("epochs_comparison.png", results_dir)


def plot_all(neuro_results, normal_results, task_names, results_dir=DEFAULT_RESULTS_DIR,
             memory_results=None, equal_compute_results=None):
    """Runs all plotting functions to create all result plots"""
    plot_training_curves(neuro_results, normal_results, task_names, results_dir, memory_results)
    plot_post_training_accuracy(neuro_results, normal_results, task_names, results_dir)
    plot_checkpoint_accuracy(neuro_results, normal_results, task_names, results_dir, memory_results)
    plot_average_accuracy(neuro_results, normal_results, task_names, results_dir, memory_results, equal_compute_results)
    plot_forgetting(neuro_results, normal_results, task_names, results_dir, memory_results, equal_compute_results)
    plot_delta_heatmap(neuro_results, normal_results, task_names, results_dir)
    for cp in ["checkpoint_2", "checkpoint_3", "checkpoint_4"]:
        plot_forgetting_heatmap(neuro_results, normal_results, task_names, checkpoint=cp, results_dir=results_dir)
    for cp in ["post_training", "checkpoint_1", "checkpoint_2", "checkpoint_3", "checkpoint_4"]:
        plot_accuracy_heatmap(neuro_results, task_names, checkpoint=cp, results_dir=results_dir)
