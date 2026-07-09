import math
import os
import shutil
import subprocess

import numpy as np
import torch
from einops import rearrange
from loguru import logger
from torch.utils.data import DataLoader, RandomSampler, TensorDataset

from model.rpacq import RPACQ
from utils.metric import Metric


class SimpleDataset(TensorDataset):
    def __init__(self, *tensors):
        super().__init__(*tensors)

    def get_data(self):
        return self.tensors


def _to_one_hot(labels: np.ndarray, num_classes: int) -> np.ndarray:
    labels = labels.astype(np.int64)
    return np.eye(num_classes, dtype=np.float32)[labels]


def _bytes_to_gib(num_bytes: int) -> float:
    return float(num_bytes) / (1024.0 ** 3)


def _classify_cuda_failure(error_text: str) -> str:
    text = error_text.lower()
    if "out of memory" in text:
        return "oom_or_memory_pressure"
    if (
        "device-side assert triggered" in text
        or "illegal memory access" in text
        or "misaligned address" in text
        or "an illegal memory access was encountered" in text
    ):
        return "kernel_or_indexing_bug"
    if "unknown error" in text or "unspecified launch failure" in text or "driver shutting down" in text:
        return "likely_gpu_reset_or_contention"
    return "unknown_cuda_failure"


def _collect_cuda_diagnostics(device: str) -> dict:
    diagnostics = {"device_arg": str(device), "cuda_available": torch.cuda.is_available()}
    if not torch.cuda.is_available():
        return diagnostics

    dev = torch.device(device)
    dev_idx = dev.index if dev.index is not None else torch.cuda.current_device()
    diagnostics["device_index"] = dev_idx
    diagnostics["device_name"] = torch.cuda.get_device_name(dev_idx)
    diagnostics["device_count"] = torch.cuda.device_count()

    try:
        free_b, total_b = torch.cuda.mem_get_info(dev_idx)
        diagnostics["free_gib"] = round(_bytes_to_gib(free_b), 3)
        diagnostics["total_gib"] = round(_bytes_to_gib(total_b), 3)
    except RuntimeError as exc:
        diagnostics["mem_get_info_error"] = str(exc)

    try:
        diagnostics["allocated_gib"] = round(_bytes_to_gib(torch.cuda.memory_allocated(dev_idx)), 3)
        diagnostics["reserved_gib"] = round(_bytes_to_gib(torch.cuda.memory_reserved(dev_idx)), 3)
        diagnostics["max_allocated_gib"] = round(_bytes_to_gib(torch.cuda.max_memory_allocated(dev_idx)), 3)
        diagnostics["max_reserved_gib"] = round(_bytes_to_gib(torch.cuda.max_memory_reserved(dev_idx)), 3)
    except RuntimeError as exc:
        diagnostics["memory_stats_error"] = str(exc)

    return diagnostics


def _query_nvidia_compute_apps() -> str | None:
    nvidia_smi_cmd = shutil.which("nvidia-smi")
    if nvidia_smi_cmd is None and os.path.exists("/usr/lib/wsl/lib/nvidia-smi"):
        nvidia_smi_cmd = "/usr/lib/wsl/lib/nvidia-smi"
    if nvidia_smi_cmd is None:
        return "[nvidia-smi unavailable in PATH and /usr/lib/wsl/lib]"

    try:
        result = subprocess.run(
            [
                nvidia_smi_cmd,
                "--query-compute-apps=pid,process_name,used_memory,gpu_uuid",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
    except FileNotFoundError:
        return f"[nvidia-smi binary not found at runtime: {nvidia_smi_cmd}]"
    except subprocess.SubprocessError as exc:
        return f"[nvidia-smi query raised error] {exc}"

    if result.returncode != 0:
        stderr = result.stderr.strip()
        return f"[nvidia-smi query failed rc={result.returncode}] {stderr}" if stderr else None

    output = result.stdout.strip()
    return output if output else None


@torch.no_grad()
def _evaluate(model: RPACQ, dataloader: DataLoader, device: str) -> float:
    model.eval()
    features, labels = dataloader.dataset.get_data()
    if labels.dim() > 1:
        labels_np = np.argmax(labels.numpy(), axis=1)
    else:
        labels_np = labels.numpy()

    y_preds = model.predict(features.to(device)).cpu().numpy()
    acc = np.sum(y_preds == labels_np) / len(labels_np)
    return acc * 100.0


def train(
    model: RPACQ,
    metric: Metric,
    train_data: np.ndarray,
    train_labels: np.ndarray,
    test_data: np.ndarray,
    test_labels: np.ndarray,
    batch_size: int,
    num_classes: int,
    device: str,
    epochs: int,
    subject_id: int,
    session_id: int,
    learning_rate: float,
    early_stop_patience: int = 0,
    xconf_ramp_epochs: int = 200,
    weight_decay: float = 1e-5,
):
    logger.info("len of train data: {}, len of test data: {}", len(train_data), len(test_data))

    train_data = train_data.astype(np.float32)
    test_data = test_data.astype(np.float32)

    train_shape = train_data.shape
    test_shape = test_data.shape
    train_flat = train_data.reshape(train_shape[0], -1)
    test_flat = test_data.reshape(test_shape[0], -1)
    train_data = train_flat.reshape(train_shape)
    test_data = test_flat.reshape(test_shape)

    if model.use_gcn:
        train_data = rearrange(train_data, "sample chan feature -> sample feature chan")
        test_data = rearrange(test_data, "sample chan feature -> sample feature chan")
    else:
        train_data = train_data.reshape(train_data.shape[0], -1)
        test_data = test_data.reshape(test_data.shape[0], -1)

    train_label_oh = _to_one_hot(train_labels, num_classes)
    test_labels = test_labels.astype(np.int64)

    source_dataset = SimpleDataset(torch.from_numpy(train_data), torch.from_numpy(train_label_oh))
    target_dataset = SimpleDataset(torch.from_numpy(test_data), torch.from_numpy(test_labels))

    source_loader = DataLoader(
        source_dataset,
        sampler=RandomSampler(source_dataset),
        batch_size=batch_size,
        num_workers=4,
        drop_last=True,
    )
    target_loader = DataLoader(
        target_dataset,
        sampler=RandomSampler(target_dataset),
        batch_size=batch_size,
        num_workers=4,
        drop_last=True,
    )

    optimizer_params = model.get_parameters()
    optimizer = torch.optim.RMSprop(
        optimizer_params,
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    best_acc = 0.0
    stop = 0

    for epoch in range(epochs):
        model.train()

        n_batch = min(len(source_loader), len(target_loader))
        if n_batch <= 0:
            logger.warning(
                "No valid RPACQ batch in epoch {}. Check batch_size={} with train/test sizes.",
                epoch + 1,
                batch_size,
            )
            break

        source_iter = iter(source_loader)
        target_iter = iter(target_loader)

        loss_clf = 0.0
        loss_transfer = 0.0
        loss_cluster = 0.0
        loss_p = 0.0
        loss_xconf_acc = 0.0

        diag = {"pool_size": 0.0, "M_t_offdiag_max": 0.0, "agree_rate": 0.0}

        for batch_idx in range(n_batch):
            try:
                src_data, src_label = next(source_iter)
            except StopIteration:
                source_iter = iter(source_loader)
                src_data, src_label = next(source_iter)

            try:
                tgt_data, _ = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                tgt_data, _ = next(target_iter)

            src_data, src_label = src_data.to(device), src_label.to(device)
            tgt_data = tgt_data.to(device)

            try:
                losses, diag = model(src_data, tgt_data, src_label, epoch, epochs)

                lam1 = 2.0 * (2.0 / (1.0 + math.exp(-epoch / max(1, epochs))) - 1.0)
                lam2 = 0.5
                lam3 = 0.2 * min(1.0, epoch / max(1, xconf_ramp_epochs))

                total_loss = (
                    losses["src_ce"]
                    + losses["dann"]
                    + lam1 * losses["tgt_ce"]
                    + lam2 * losses["tri"]
                    + lam3 * losses["xconf"]
                )

                if total_loss.requires_grad:
                    optimizer.zero_grad()
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                loss_clf += losses["src_ce"].detach().item()
                loss_transfer += losses["dann"].detach().item()
                loss_cluster += losses["tgt_ce"].detach().item()
                loss_p += losses["tri"].detach().item()
                loss_xconf_acc += losses["xconf"].detach().item()

            except RuntimeError as exc:
                error_text = str(exc)
                if "cuda" not in error_text.lower():
                    raise

                failure_type = _classify_cuda_failure(error_text)
                if torch.cuda.is_available():
                    try:
                        torch.cuda.synchronize()
                    except RuntimeError:
                        pass

                diagnostics = _collect_cuda_diagnostics(device)
                compute_apps = _query_nvidia_compute_apps()

                mem_info_error = str(diagnostics.get("mem_get_info_error", "")).lower()
                if failure_type == "kernel_or_indexing_bug" and "unknown error" in error_text.lower():
                    failure_type = "likely_gpu_reset_or_contention"
                if "unknown error" in mem_info_error and "device-side assert triggered" not in error_text.lower():
                    failure_type = "likely_gpu_reset_or_contention"

                logger.exception(
                    "CUDA failure at epoch {}/{} batch {}/{} while training subject {} session {}",
                    epoch + 1, epochs, batch_idx + 1, n_batch, subject_id, session_id,
                )
                logger.error("CUDA failure class: {}", failure_type)
                logger.error("CUDA diagnostics: {}", diagnostics)
                if compute_apps is not None:
                    logger.error("Active GPU compute processes at failure:\n{}", compute_apps)

                if failure_type == "likely_gpu_reset_or_contention":
                    logger.error(
                        "Likely cross-process GPU contention/reset. "
                        "Isolate workloads by GPU or reduce memory pressure."
                    )
                elif failure_type == "kernel_or_indexing_bug":
                    logger.error(
                        "Likely model/kernel issue. "
                        "Re-run with CUDA_LAUNCH_BLOCKING=1 to identify the failing operation."
                    )
                else:
                    logger.error(
                        "Unclassified CUDA failure. "
                        "Re-run with CUDA_LAUNCH_BLOCKING=1 and inspect the per-process GPU snapshot."
                    )

                raise RuntimeError(
                    f"CUDA training failure ({failure_type}) at epoch {epoch + 1} batch {batch_idx + 1}. "
                    "See logs for diagnostics and process snapshot."
                ) from exc

        target_acc = _evaluate(model, target_loader, device)
        metric.update(subject_id, session_id, target_acc / 100.0)

        if target_acc > best_acc:
            best_acc = target_acc
            stop = 0
        else:
            stop += 1

        if (epoch + 1) % 50 == 0 or epoch == 0:
            source_acc = _evaluate(model, source_loader, device)
            logger.info(
                "Epoch {}/{} | src_ce: {:.4f}, dann: {:.4f}, tgt_ce: {:.4f}, "
                "tri: {:.4f}, xconf: {:.4f}, source_acc: {:.4f}, target_acc: {:.4f}, best_acc: {:.4f}",
                epoch + 1, epochs,
                loss_clf / n_batch, loss_transfer / n_batch, loss_cluster / n_batch,
                loss_p / n_batch, loss_xconf_acc / n_batch,
                source_acc, target_acc, best_acc,
            )
            logger.info(
                "Epoch {}/{} | pool_size: {}, agree_rate: {:.4f}, M_t_offdiag_max: {:.4f}",
                epoch + 1, epochs,
                int(diag["pool_size"]), diag["agree_rate"], diag["M_t_offdiag_max"],
            )

        if best_acc >= (100.0 - 1e-1) or (early_stop_patience > 0 and stop >= early_stop_patience):
            logger.info("Early stop at epoch {} with best target acc {:.4f}", epoch + 1, best_acc)
            break
