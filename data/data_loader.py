import yaml
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer

TOKENIZER = BertTokenizer.from_pretrained("google/bert_uncased_L-2_H-128_A-2")


class TaskDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def tokenize(texts):
    return TOKENIZER(texts, padding=True, truncation=True, max_length=128)


def make_dataloader(texts, labels, batch_size=32, shuffle=False):
    encodings = tokenize(list(texts))
    dataset = TaskDataset(encodings, list(labels))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def load_all_tasks(config_path="config.yaml"):
    config = load_config(config_path)
    seed = config["seed"]
    train_size = config["train_size"]
    memory_size = config["memory_size"]
    test_size = config["test_size"]

    tasks = {}
    for task in config["tasks"]:
        name = task["name"]
        num_classes = task["num_classes"]

        raw = load_dataset(name)
        train_split = raw["train"].shuffle(seed=seed)
        test_split = raw["test"].shuffle(seed=seed)

        train_split = train_split.select(range(train_size + memory_size))
        test_split = test_split.select(range(test_size))

        text_col = task["text_col"]
        label_col = task["label_col"]

        train_texts = train_split.select(range(train_size))[text_col]
        train_labels = train_split.select(range(train_size))[label_col]
        memory_texts = train_split.select(range(train_size, train_size + memory_size))[text_col]
        memory_labels = train_split.select(range(train_size, train_size + memory_size))[label_col]
        test_texts = test_split[text_col]
        test_labels = test_split[label_col]

        tasks[name] = {
            "train_loader": make_dataloader(train_texts, train_labels, shuffle=True),
            "memory_loader": make_dataloader(memory_texts, memory_labels, shuffle=True),
            "test_loader": make_dataloader(test_texts, test_labels, shuffle=False),
            "num_classes": num_classes,
        }

    return tasks
