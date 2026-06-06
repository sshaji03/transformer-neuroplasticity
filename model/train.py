import torch
from torch.optim import AdamW
from evaluate.evaluate import evaluate_all_tasks
from model.utils import train_epoch


def normal_train(model, tasks, config):
    """train on each task sequentially"""
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


def equal_compute_memory_train(model, tasks, config):
    """Equal-compute memory replay baseline: train on the same number of total epochs as the neuroplasticity loop"""
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)

    epochs = config["epochs"]
    lr = config["learning_rate"]
    task_names = list(tasks.keys())
    results = {}

    for task_idx, task_name in enumerate(task_names):
        train_loader = tasks[task_name]["train_loader"]
        memory_loaders = [tasks[task_names[i]]["memory_loader"] for i in range(task_idx)]
        prev_task_names = task_names[:task_idx]
        test_loaders = {task_names[i]: tasks[task_names[i]]["test_loader"] for i in range(task_idx + 1)}

        optimizer = AdamW(model.parameters(), lr=lr)
        training_curves = []

        for epoch in range(epochs):
            model.set_task(task_name)
            metrics = train_epoch(model, train_loader, optimizer, device)
            training_curves.append(metrics)
            print(f"[EqCompute] Task {task_idx + 1} [{task_name}] Epoch {epoch + 1}/{epochs} "
                  f"loss: {metrics['loss']:.4f} acc: {metrics['accuracy']:.4f}")

        for epoch in range(epochs):
            model.set_task(task_name)
            train_epoch(model, train_loader, optimizer, device)

        for epoch in range(epochs):
            for prev_task_name, prev_loader in zip(prev_task_names, memory_loaders):
                model.set_task(prev_task_name)
                train_epoch(model, prev_loader, optimizer, device)

        for _ in range(2):
            for epoch in range(epochs):
                model.set_task(task_name)
                train_epoch(model, train_loader, optimizer, device)
                for prev_task_name, prev_loader in zip(prev_task_names, memory_loaders):
                    model.set_task(prev_task_name)
                    train_epoch(model, prev_loader, optimizer, device)

        results[task_name] = {
            "training_curves": training_curves,
            "post_training": evaluate_all_tasks(model, test_loaders, device),
        }

    return results


def memory_train(model, tasks, config):
    """memory replay baseline: train on current task data and memory data at each sequential step"""
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)

    epochs = config["epochs"]
    lr = config["learning_rate"]
    task_names = list(tasks.keys())

    results = {}

    for task_idx, task_name in enumerate(task_names):
        train_loader = tasks[task_name]["train_loader"]
        memory_loaders = [tasks[task_names[i]]["memory_loader"] for i in range(task_idx)]
        prev_task_names = task_names[:task_idx]
        test_loaders = {task_names[i]: tasks[task_names[i]]["test_loader"] for i in range(task_idx + 1)}

        optimizer = AdamW(model.parameters(), lr=lr)
        training_curves = []

        for epoch in range(epochs):
            model.set_task(task_name)
            metrics = train_epoch(model, train_loader, optimizer, device)
            training_curves.append(metrics)
            print(f"[Memory] Task {task_idx + 1} [{task_name}] Epoch {epoch + 1}/{epochs} loss: {metrics['loss']:.4f} acc: {metrics['accuracy']:.4f}")
            for prev_task_name, prev_loader in zip(prev_task_names, memory_loaders):
                model.set_task(prev_task_name)
                train_epoch(model, prev_loader, optimizer, device)

        results[task_name] = {
            "training_curves": training_curves,
            "post_training": evaluate_all_tasks(model, test_loaders, device),
        }

    return results
