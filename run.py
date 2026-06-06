import os
import torch
from data.data_loader import load_all_tasks, load_config
from model.model import BertWithTaskHeads
from model.neuroplasticity_train import neuroplasticity_train
from model.train import normal_train, memory_train, equal_compute_memory_train
from experiment.graphs import plot_all, plot_k_comparison, plot_epochs_comparison


def make_model(config):
    model = BertWithTaskHeads()
    for task in config["tasks"]:
        model.add_task_head(task["name"], task["num_classes"])
    return model


def main():
    config = load_config()
    tasks = load_all_tasks()
    task_names = [task["name"] for task in config["tasks"]]

    k_values = config["cwi_top_k"]
    if isinstance(k_values, (int, float)):
        k_values = [k_values]

    epoch_values = config["epochs"]
    if isinstance(epoch_values, int):
        epoch_values = [epoch_values]

    # all_neuro_results[epochs][k]         = neuro_results
    # all_normal_results[epochs]            = normal_results
    # all_memory_results[epochs]            = memory_results (basic replay)
    # all_equal_compute_results[epochs]     = equal_compute_results (compute-matched replay)
    all_neuro_results = {}
    all_normal_results = {}
    all_memory_results = {}
    all_equal_compute_results = {}

    for epochs in epoch_values:
        epoch_config = {**config, "epochs": epochs}

        print(f"\n=== Normal training  epochs={epochs} ===")
        torch.manual_seed(config["seed"])
        normal_model = make_model(config)
        normal_results = normal_train(normal_model, tasks, epoch_config)
        all_normal_results[epochs] = normal_results

        print(f"\n=== Memory replay training  epochs={epochs} ===")
        torch.manual_seed(config["seed"])
        memory_model = make_model(config)
        memory_results = memory_train(memory_model, tasks, epoch_config)
        all_memory_results[epochs] = memory_results

        print(f"\n=== Equal-compute memory replay training  epochs={epochs} ===")
        torch.manual_seed(config["seed"])
        equal_compute_model = make_model(config)
        equal_compute_results = equal_compute_memory_train(equal_compute_model, tasks, epoch_config)
        all_equal_compute_results[epochs] = equal_compute_results

        all_neuro_results[epochs] = {}
        for k in k_values:
            print(f"\n=== Neuroplasticity training  epochs={epochs}  k={k} ===")
            run_config = {**epoch_config, "cwi_top_k": k}
            results_dir = os.path.join("results_memory", f"epochs_{epochs}", f"k_{k}")
            checkpoint_dir = os.path.join("checkpoints", f"epochs_{epochs}", f"k_{k}")

            torch.manual_seed(config["seed"])
            neuro_model = make_model(config)
            neuro_results = neuroplasticity_train(
                neuro_model, tasks, run_config,
                checkpoint_dir=checkpoint_dir,
            )
            all_neuro_results[epochs][k] = neuro_results
            plot_all(neuro_results, normal_results, task_names, results_dir=results_dir,
                     memory_results=memory_results, equal_compute_results=equal_compute_results)

        # k-level summary for this epoch count
        plot_k_comparison(
            all_neuro_results[epochs], normal_results, task_names,
            results_dir=os.path.join("results_memory", f"epochs_{epochs}"),
            memory_results=memory_results,
            equal_compute_results=equal_compute_results,
        )

    # Epoch-level summary across all epoch values
    print("\n=== Generating epoch comparison plots ===")
    plot_epochs_comparison(
        all_neuro_results, all_normal_results, task_names, k_values,
        results_dir="results_memory",
        all_memory_results=all_memory_results,
        all_equal_compute_results=all_equal_compute_results,
    )


if __name__ == "__main__":
    main()
