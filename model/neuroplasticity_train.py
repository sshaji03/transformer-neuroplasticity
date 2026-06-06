import copy
import os
import torch
from torch.optim import AdamW
from evaluate.evaluate import evaluate_all_tasks
from model.utils import train_epoch

CHECKPOINT_DIR = "checkpoints"


def save_checkpoint(state_dict, task_name, checkpoint_num, checkpoint_dir=CHECKPOINT_DIR):
    """Saves a checkpoint model"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    safe_name = task_name.replace("/", "_")
    path = os.path.join(checkpoint_dir, f"{safe_name}_cp{checkpoint_num}.pt")
    torch.save(state_dict, path)


def is_backbone(name):
    """used to make sure only bert parameters are used in CWI scoring not task head"""
    return name.startswith("bert.")


def compute_full_gradients(model, loader, device):
    """used to compute average gradient over previous tasks"""
    model.train()
    model.zero_grad()
    n_batches = len(loader)
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        (outputs.loss / n_batches).backward()


def save_gradients(model):
    """saves current gradient so can be used for future CWI calculations"""
    return {
        name: param.grad.abs().clone()
        for name, param in model.named_parameters()
        if param.requires_grad and param.grad is not None and is_backbone(name)
    }


def normalize_tensor(t):
    """normalizes the values being summed for CWI score"""
    return t / (t.mean() + 1e-8)


def compute_cwi_scores(model, saved_gradients, config):
    """Computes CWI score for every backbond parameter, based on weight magnitude, current gradient, and
    average of other gradients"""

    # read in from config file
    alpha = config["cwi_magnitude_weight"]
    beta = config["cwi_current_grad_weight"]
    gamma = config["cwi_prev_grad_weight"]

    magnitude = {name: param.data.abs() for name, param in model.named_parameters() if param.requires_grad and is_backbone(name)}
    current_grads = {
        name: param.grad.abs()
        for name, param in model.named_parameters()
        if param.requires_grad and param.grad is not None and is_backbone(name)
    }

    if saved_gradients:
        prev_grads = {
            name: torch.stack([saved_gradients[t][name] for t in saved_gradients]).mean(0)
            for name in current_grads
        }
    else:
        prev_grads = {name: torch.zeros_like(current_grads[name]) for name in current_grads}

    cwi_scores = {
        name: alpha * normalize_tensor(magnitude[name])
               + beta * normalize_tensor(current_grads[name])
               + gamma * normalize_tensor(prev_grads[name])
        for name in current_grads
    }
    return cwi_scores


def get_importance_mask(cwi_scores, top_k):
    """uses topk to create a mask that has the important weight masks and unimportant weights mask"""
    masks = {}
    for name, scores in cwi_scores.items():
        flat = scores.flatten()
        k = max(1, int(len(flat) * top_k))
        threshold = torch.topk(flat, k).values.min()
        masks[name] = scores >= threshold
    return masks


def apply_mask_to_state(state_dict, importance_mask, keep_important):
    """Zero out any weights in order to train the networks separately"""
    masked_state = copy.deepcopy(state_dict)
    for name, mask in importance_mask.items():
        if keep_important:
            masked_state[name] = masked_state[name] * mask.float()
        else:
            masked_state[name] = masked_state[name] * (~mask).float()
    return masked_state


def register_grad_hooks(model, importance_mask, train_important):
    """Attach gradient hooks for freezing a subnetwork during training to train separately"""
    hooks = []
    for name, param in model.named_parameters():
        if name in importance_mask:
            mask = importance_mask[name]
            if train_important:
                hook = param.register_hook(lambda grad, m=mask: grad * m.float())
            else:
                hook = param.register_hook(lambda grad, m=mask: grad * (~m).float())
            hooks.append(hook)
    return hooks


def remove_hooks(hooks):
    """remove hooks after subnetwork training is done"""
    for hook in hooks:
        hook.remove()


def merge_states(important_state, unimportant_state, importance_mask):
    """Merge weights together after training two subnetworks separately"""
    merged = copy.deepcopy(important_state)
    for name, mask in importance_mask.items():
        merged[name] = important_state[name] * mask.float() + unimportant_state[name] * (~mask).float()
    return merged


def reset_unimportant_weights(state_dict, initial_state, importance_mask):
    """reset the unimportant weights to the pretrained BERT baseline (active forgetting)"""
    reset_state = copy.deepcopy(state_dict)
    for name, mask in importance_mask.items():
        reset_state[name] = state_dict[name] * mask.float() + initial_state[name] * (~mask).float()
    return reset_state


def retrain_full(state_dict, model, task_name, prev_task_names, train_loader, memory_loaders, optimizer, epochs, device):
    """Fine tune a model using task and memory data"""
    model.load_state_dict(state_dict)
    for epoch in range(epochs):
        model.set_task(task_name)
        train_epoch(model, train_loader, optimizer, device)
        for prev_task_name, prev_loader in zip(prev_task_names, memory_loaders):
            model.set_task(prev_task_name)
            train_epoch(model, prev_loader, optimizer, device)
    return model.state_dict()


def neuroplasticity_train(model, tasks, config, checkpoint_dir=CHECKPOINT_DIR):
    """Main training loop to implement the different checkpoints and separated training"""
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    initial_state = copy.deepcopy(model.state_dict())

    epochs = config["epochs"]
    lr = config["learning_rate"]
    top_k = config["cwi_top_k"]
    task_names = list(tasks.keys())

    saved_gradients = {}
    checkpoints = {}
    results = {}

    for task_idx, task_name in enumerate(task_names):
        train_loader = tasks[task_name]["train_loader"]
        memory_loaders = [tasks[task_names[i]]["memory_loader"] for i in range(task_idx)]
        prev_task_names = task_names[:task_idx]
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

        compute_full_gradients(model, train_loader, device)
        cwi_scores = compute_cwi_scores(model, saved_gradients, config)
        importance_mask = get_importance_mask(cwi_scores, top_k)
        saved_gradients[task_name] = save_gradients(model)

        # CP1: important weights only
        pre_split_state = copy.deepcopy(model.state_dict())
        checkpoints[task_name] = {}
        checkpoints[task_name]["checkpoint_1"] = apply_mask_to_state(pre_split_state, importance_mask, keep_important=True)
        save_checkpoint(checkpoints[task_name]["checkpoint_1"], task_name, 1, checkpoint_dir)
        cp1_model = copy.deepcopy(model)
        cp1_model.load_state_dict(checkpoints[task_name]["checkpoint_1"])
        results[task_name]["checkpoint_1"] = evaluate_all_tasks(cp1_model, test_loaders, device)

        # retrain important weights on task data
        important_model = copy.deepcopy(model)
        important_model.set_task(task_name)
        hooks = register_grad_hooks(important_model, importance_mask, train_important=True)
        optimizer_imp = AdamW(important_model.parameters(), lr=lr)
        for epoch in range(epochs):
            train_epoch(important_model, train_loader, optimizer_imp, device)
        remove_hooks(hooks)

        # train unimportant weights on memory data
        unimportant_model = copy.deepcopy(model)
        hooks = register_grad_hooks(unimportant_model, importance_mask, train_important=False)
        if memory_loaders:
            optimizer_unimp = AdamW(unimportant_model.parameters(), lr=lr)
            for epoch in range(epochs):
                for prev_task_name, prev_loader in zip(prev_task_names, memory_loaders):
                    unimportant_model.set_task(prev_task_name)
                    train_epoch(unimportant_model, prev_loader, optimizer_unimp, device)
        remove_hooks(hooks)

        # CP2: merge important and unimportant subnetworks together
        checkpoints[task_name]["checkpoint_2"] = merge_states(
            important_model.state_dict(), unimportant_model.state_dict(), importance_mask
        )
        save_checkpoint(checkpoints[task_name]["checkpoint_2"], task_name, 2, checkpoint_dir)
        cp2_model = copy.deepcopy(model)
        cp2_model.load_state_dict(checkpoints[task_name]["checkpoint_2"])
        results[task_name]["checkpoint_2"] = evaluate_all_tasks(cp2_model, test_loaders, device)

        # CP3: reset unimportant weights CP1, retrain on full data
        cp3_start = reset_unimportant_weights(checkpoints[task_name]["checkpoint_1"], initial_state, importance_mask)
        cp3_model = copy.deepcopy(model)
        optimizer_cp3 = AdamW(cp3_model.parameters(), lr=lr)
        checkpoints[task_name]["checkpoint_3"] = retrain_full(
            cp3_start, cp3_model, task_name, prev_task_names, train_loader, memory_loaders, optimizer_cp3, epochs, device
        )
        save_checkpoint(checkpoints[task_name]["checkpoint_3"], task_name, 3, checkpoint_dir)
        cp3_model.load_state_dict(checkpoints[task_name]["checkpoint_3"])
        results[task_name]["checkpoint_3"] = evaluate_all_tasks(cp3_model, test_loaders, device)

        # CP4: reset unimportant weights CP2, retrain on full data
        cp4_start = reset_unimportant_weights(checkpoints[task_name]["checkpoint_2"], initial_state, importance_mask)
        cp4_model = copy.deepcopy(model)
        optimizer_cp4 = AdamW(cp4_model.parameters(), lr=lr)
        checkpoints[task_name]["checkpoint_4"] = retrain_full(
            cp4_start, cp4_model, task_name, prev_task_names, train_loader, memory_loaders, optimizer_cp4, epochs, device
        )
        save_checkpoint(checkpoints[task_name]["checkpoint_4"], task_name, 4, checkpoint_dir)
        cp4_model.load_state_dict(checkpoints[task_name]["checkpoint_4"])
        results[task_name]["checkpoint_4"] = evaluate_all_tasks(cp4_model, test_loaders, device)

    return results
