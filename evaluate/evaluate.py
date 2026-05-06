import torch


def evaluate_model(model, loader, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()
            preds = outputs.logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return {
        "loss": total_loss / len(loader),
        "accuracy": correct / total,
    }


def evaluate_all_tasks(model, test_loaders, device):
    results = {}
    for task_name, loader in test_loaders.items():
        model.set_task(task_name)
        results[task_name] = evaluate_model(model, loader, device)
    return results
