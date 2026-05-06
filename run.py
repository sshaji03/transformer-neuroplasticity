import copy
from data.data_loader import load_all_tasks, load_config
from model.model import BertWithTaskHeads
from model.neuroplasticity_train import neuroplasticity_train
from model.train import normal_train
from experiment.graphs import plot_all


def main():
    config = load_config()
    tasks = load_all_tasks()
    task_names = [task["name"] for task in config["tasks"]]

    neuro_model = BertWithTaskHeads()
    for task in config["tasks"]:
        neuro_model.add_task_head(task["name"], task["num_classes"])

    normal_model = BertWithTaskHeads()
    for task in config["tasks"]:
        normal_model.add_task_head(task["name"], task["num_classes"])

    neuro_results = neuroplasticity_train(neuro_model, tasks, config)
    normal_results = normal_train(normal_model, tasks, config)

    plot_all(neuro_results, normal_results, task_names)


if __name__ == "__main__":
    main()
