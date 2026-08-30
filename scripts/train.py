"""Train the conditional computation model from a YAML configuration."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch_optimizer
import yaml
from torch.utils.data import DataLoader, TensorDataset

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from ocean_ccm.data import (
    NormalizationStats,
    ProcessedDataset,
    get_feature_layout,
    load_mat_dataset,
    split_indices,
)
from ocean_ccm.losses import branch_losses
from ocean_ccm.model import ConditionalModel


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def as_loader(
    data: ProcessedDataset,
    *,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    tensors = [
        torch.from_numpy(data.spatial),
        torch.from_numpy(data.vector),
        torch.from_numpy(data.previous_pressure),
        torch.from_numpy(data.eddy_signal),
        torch.from_numpy(data.targets),
    ]
    return DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=shuffle)


def batch_loss(
    outputs: dict,
    targets: torch.Tensor,
    non_eddy_loss,
    eddy_loss,
) -> tuple[torch.Tensor, torch.Tensor]:
    zero = targets.sum() * 0.0
    normal = zero
    outlier = zero
    if "normal" in outputs:
        group = outputs["normal"]
        normal = non_eddy_loss(group["outputs"], targets[group["indices"]])
    if "outlier" in outputs:
        group = outputs["outlier"]
        outlier = eddy_loss(group["outputs"], targets[group["indices"]])
    return normal, outlier


def evaluate_loss(model, loader, device, non_eddy_loss, eddy_loss) -> float:
    model.eval()
    total = 0.0
    with torch.no_grad():
        for batch in loader:
            spatial, vector, pressure, signal, target = [value.to(device) for value in batch]
            outputs = model(spatial, vector, pressure, signal)
            normal, outlier = batch_loss(outputs, target, non_eddy_loss, eddy_loss)
            total += float(normal + outlier)
    return total / max(len(loader), 1)


def main() -> None:
    arguments = parse_arguments()
    configuration = yaml.safe_load(arguments.config.read_text(encoding="utf-8"))
    seed = int(configuration["seed"])
    seed_everything(seed)

    data_configuration = configuration["data"]
    layout = get_feature_layout(data_configuration["feature_layout"])
    archived_train = data_configuration.get("train_mat")
    archived_validation = data_configuration.get("validation_mat")
    if bool(archived_train) != bool(archived_validation):
        raise ValueError("data.train_mat and data.validation_mat must be supplied together")

    if archived_train:
        raw_train = load_mat_dataset(Path(archived_train))
        raw_validation = load_mat_dataset(Path(archived_validation))
        train_indices = np.arange(raw_train.size)
        validation_indices = np.arange(raw_validation.size)
    else:
        paths = [Path(data_configuration["primary_mat"])]
        supplement = data_configuration.get("supplement_mat")
        if supplement:
            paths.append(Path(supplement))
        raw = load_mat_dataset(paths)
        train_indices, validation_indices = split_indices(
            raw.size,
            train_fraction=float(data_configuration["train_fraction"]),
            seed=seed,
        )
        raw_train = raw
        raw_validation = raw

    stats = NormalizationStats.fit(raw_train, train_indices, layout=layout)
    vector_index = tuple(data_configuration.get("vector_grid_index", [3, 3]))
    missing_value = float(data_configuration.get("missing_value", -999.0))
    missing_value_policy = data_configuration.get("missing_value_policy", "sentinel")
    train_data = stats.transform(
        raw_train,
        train_indices,
        vector_grid_index=vector_index,
        missing_value=missing_value,
        missing_value_policy=missing_value_policy,
    )
    validation_data = stats.transform(
        raw_validation,
        validation_indices,
        vector_grid_index=vector_index,
        missing_value=missing_value,
        missing_value_policy=missing_value_policy,
    )

    training_configuration = configuration["training"]
    train_loader = as_loader(
        train_data,
        batch_size=int(training_configuration["batch_size"]),
        shuffle=True,
    )
    validation_loader = as_loader(
        validation_data,
        batch_size=int(training_configuration["batch_size"]),
        shuffle=False,
    )

    model_configuration = configuration["model"]
    model = ConditionalModel(
        input_shape_spatial=(len(layout.spatial_indices), 8, 8),
        num_depths=int(model_configuration["num_depths"]),
        drop=float(model_configuration["dropout"]),
        mask_mode=model_configuration.get("mask_mode", "legacy"),
        pool_mode=model_configuration.get("pool_mode", "max"),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    normal_parameters = [
        parameter for name, parameter in model.named_parameters() if name.startswith("normal_")
    ]
    outlier_parameters = [
        parameter for name, parameter in model.named_parameters() if name.startswith("outlier_")
    ]
    learning_rate = float(training_configuration["learning_rate"])
    optimizer_normal = torch_optimizer.RAdam(normal_parameters, lr=learning_rate)
    optimizer_outlier = torch_optimizer.RAdam(outlier_parameters, lr=learning_rate)

    non_eddy_loss, eddy_loss = branch_losses(
        training_configuration["loss_profile"],
        non_eddy_gamma=float(training_configuration.get("non_eddy_gamma", 1.5)),
        eddy_gamma=float(training_configuration.get("eddy_gamma", 2.0)),
    )

    output_directory = Path(training_configuration["output_dir"])
    output_directory.mkdir(parents=True, exist_ok=True)
    stats.save(output_directory / "stats.npz")
    np.savez(
        output_directory / "split_indices.npz",
        train=train_indices,
        validation=validation_indices,
    )
    (output_directory / "resolved_config.yaml").write_text(
        yaml.safe_dump(configuration, sort_keys=False), encoding="utf-8"
    )

    best_validation = float("inf")
    epochs = int(training_configuration["epochs"])
    for epoch in range(1, epochs + 1):
        model.train()
        running_normal = 0.0
        running_outlier = 0.0
        for batch in train_loader:
            spatial, vector, pressure, signal, target = [value.to(device) for value in batch]
            optimizer_normal.zero_grad()
            optimizer_outlier.zero_grad()
            outputs = model(spatial, vector, pressure, signal)
            normal, outlier = batch_loss(outputs, target, non_eddy_loss, eddy_loss)
            (normal + outlier).backward()
            optimizer_normal.step()
            optimizer_outlier.step()
            running_normal += float(normal.detach())
            running_outlier += float(outlier.detach())

        validation_loss = evaluate_loss(
            model, validation_loader, device, non_eddy_loss, eddy_loss
        )
        print(
            f"epoch={epoch:04d} "
            f"normal_loss={running_normal / len(train_loader):.6f} "
            f"eddy_loss={running_outlier / len(train_loader):.6f} "
            f"validation_loss={validation_loss:.6f}"
        )

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_normal_state_dict": optimizer_normal.state_dict(),
            "optimizer_outlier_state_dict": optimizer_outlier.state_dict(),
            "configuration": configuration,
            "validation_loss": validation_loss,
        }
        if validation_loss < best_validation:
            best_validation = validation_loss
            torch.save(checkpoint, output_directory / "best.pth")
        if epoch == epochs:
            torch.save(checkpoint, output_directory / "last.pth")


if __name__ == "__main__":
    main()

