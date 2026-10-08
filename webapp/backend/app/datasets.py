"""Preloaded torchvision datasets, importable directly -- no manual
download/extraction steps. First use downloads to webapp/data/ (cached
there afterward); the backend process must have internet access once,
the first time a given dataset is selected.
"""

from pathlib import Path

import torchvision
import torchvision.transforms as transforms

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

DATASET_SPECS = {
    "cifar10": {
        "label": "CIFAR-10",
        "input_shape": (3, 32, 32),
        "num_classes": 10,
        "class_names": [
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck",
        ],
        "cls": torchvision.datasets.CIFAR10,
    },
    "cifar100": {
        "label": "CIFAR-100",
        "input_shape": (3, 32, 32),
        "num_classes": 100,
        "class_names": None,
        "cls": torchvision.datasets.CIFAR100,
    },
    "mnist": {
        "label": "MNIST",
        "input_shape": (1, 28, 28),
        "num_classes": 10,
        "class_names": [str(i) for i in range(10)],
        "cls": torchvision.datasets.MNIST,
    },
    "fashionmnist": {
        "label": "Fashion-MNIST",
        "input_shape": (1, 28, 28),
        "num_classes": 10,
        "class_names": [
            "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
            "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
        ],
        "cls": torchvision.datasets.FashionMNIST,
    },
}

_TRANSFORM = transforms.ToTensor()


def list_datasets() -> list:
    return [
        {
            "key": key,
            "label": spec["label"],
            "input_shape": list(spec["input_shape"]),
            "num_classes": spec["num_classes"],
        }
        for key, spec in DATASET_SPECS.items()
    ]


def get_dataset_spec(key: str) -> dict:
    spec = DATASET_SPECS.get(key)
    if spec is None:
        raise ValueError(f"Unknown dataset '{key}'. Known: {', '.join(DATASET_SPECS)}")
    return spec


def load_datasets(key: str):
    """Returns (train_dataset, test_dataset), downloading to DATA_DIR on
    first use."""
    spec = get_dataset_spec(key)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cls = spec["cls"]
    train_set = cls(root=str(DATA_DIR), train=True, download=True, transform=_TRANSFORM)
    test_set = cls(root=str(DATA_DIR), train=False, download=True, transform=_TRANSFORM)
    return train_set, test_set
