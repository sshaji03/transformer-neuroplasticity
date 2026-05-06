import torch
from torch.optim import AdamW
from evaluate.evaluate import evaluate_all_tasks
from model.utils import train_epoch


def normal_train(model, tasks, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)

    epochs = config["epochs"]
    lr = config["learning_rate"]
    task_names = list(tasks.keys())

    results = {}

    for task_idx, task_name in enumerate(task_names):
        train_loader = tasks[task_name]["train_loader"]
        test_loaders = {task_names[i]: tasks[task_names[i]]["test_loader"] for i in range(task_idx + 1)}

        model.set_task(task_name)
        optimizer = AdamW(model.parameters(), lr=lr)
        training_curves = []

        for epoch in range(epochs):
            metrics = train_epoch(model, train_loader, optimizer, device)
            training_curves.append(metrics)
            print(f"Task {task_idx + 1} [{task_name}] Epoch {epoch + 1}/{epochs} loss: {metrics['loss']:.4f} acc: {metrics['accuracy']:.4f}")

        results[task_name] = {
            "training_curves": training_curves,
            "post_training": evaluate_all_tasks(model, test_loaders, device),
        }

    return results
