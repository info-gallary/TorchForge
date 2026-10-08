"""Builds a model from a canvas graph (now a DAG, not just a chain) and
trains it, pushing progress messages into a thread-safe queue as it goes.
After training, tries to accelerate the model's trailing Linear/ReLU
suffix with the real NeuroCUDA compiler (see neurocuda_bridge.py).

Training is genuinely blocking (CPU/GPU-bound PyTorch calls), so it runs
on a background thread; the FastAPI WebSocket handler in main.py drains
the queue with `run_in_executor` (so it never blocks the asyncio event
loop) and forwards each message to the browser as JSON.
"""

import queue
import threading
import time
from pathlib import Path

import torch
import torch.nn as nn

from app.datasets import get_dataset_spec, load_datasets
from app.graph import GraphError, resolve_dag
from app.layers import build_dag
from app.neurocuda_bridge import try_accelerate

STEP_LOG_INTERVAL = 20
WORKDIR_ROOT = Path(__file__).resolve().parent.parent.parent / "data" / "_runs"


def start_training(nodes: list, edges: list, config: dict) -> "queue.Queue":
    """Kicks off training on a background thread and returns the queue it
    will report progress to. Every message is a JSON-able dict with a
    "type" field: "status", "step", "epoch_end", "neurocuda", "done", or
    "error".
    """
    q: queue.Queue = queue.Queue()
    thread = threading.Thread(target=_train_worker, args=(nodes, edges, config, q), daemon=True)
    thread.start()
    return q


def _train_worker(nodes: list, edges: list, config: dict, q: "queue.Queue") -> None:
    try:
        dataset_key = config["dataset"]
        epochs = int(config.get("epochs", 3))
        batch_size = int(config.get("batch_size", 128))
        lr = float(config.get("lr", 1e-3))
        optimizer_name = config.get("optimizer", "adam")

        q.put({"type": "status", "message": "Resolving architecture..."})
        order_ids, nodes_by_id, incoming, outgoing, source_id, sink_id = resolve_dag(nodes, edges)

        spec = get_dataset_spec(dataset_key)
        model, shapes_by_node, warnings, ends_with_logsoftmax = build_dag(
            order_ids, nodes_by_id, incoming, outgoing, source_id, sink_id,
            tuple(spec["input_shape"]),
        )
        for w in warnings:
            q.put({"type": "status", "message": w})

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        q.put({"type": "status", "message": f"Training on {device}..."})

        criterion = nn.NLLLoss() if ends_with_logsoftmax else nn.CrossEntropyLoss()
        if optimizer_name == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        q.put({"type": "status", "message": f"Loading {spec['label']}..."})
        train_set, test_set = load_datasets(dataset_key)
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=batch_size, shuffle=True, num_workers=0
        )
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, num_workers=0)

        q.put({
            "type": "status",
            "message": f"{len(train_set)} train / {len(test_set)} test images, "
                       f"{sum(p.numel() for p in model.parameters())} parameters",
        })

        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            t0 = time.time()
            for step, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                output = model(images)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                if step % STEP_LOG_INTERVAL == 0:
                    q.put({
                        "type": "step",
                        "epoch": epoch,
                        "step": step,
                        "total_steps": len(train_loader),
                        "loss": loss.item(),
                    })

            val_loss, val_acc = _evaluate(model, test_loader, criterion, device)
            q.put({
                "type": "epoch_end",
                "epoch": epoch,
                "train_loss": running_loss / max(1, len(train_loader)),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "elapsed_s": time.time() - t0,
            })

        q.put({"type": "status", "message": "Trying to accelerate the trailing layers with NeuroCUDA..."})
        workdir = WORKDIR_ROOT / f"run_{int(time.time() * 1000)}"
        sample_batch = next(iter(test_loader))
        accel_result = try_accelerate(
            model, order_ids, nodes_by_id, incoming, sink_id, sample_batch, device, workdir
        )
        q.put({"type": "neurocuda", **accel_result})

        q.put({"type": "done"})

    except GraphError as e:
        q.put({"type": "error", "message": str(e)})
    except Exception as e:  # noqa: BLE001 - surface any failure to the UI, not a 500 with no detail
        q.put({"type": "error", "message": f"{type(e).__name__}: {e}"})


def _evaluate(model, loader, criterion, device) -> tuple:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            loss = criterion(output, labels)
            total_loss += loss.item() * labels.size(0)
            correct += (output.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    return total_loss / max(1, total), correct / max(1, total)
