"""Optional frontier tabular models used by the research benchmark.

The wrappers in this module deliberately keep optional imports local.  The core
package therefore remains usable without PyTorch, TabM, or ChimeraBoost, while a
frontier run still carries a serializable preprocessing and model contract.
"""

from __future__ import annotations

import gc
import math
import random
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.metrics import log_loss
from sklearn.preprocessing import QuantileTransformer

from .modeling import (
    CATEGORICAL_FEATURES,
    MODEL_FEATURES,
    NUMERIC_MODEL_FEATURES,
    CategoryCodec,
    TaskName,
    engineer_model_features,
)

# TabM's built-in categorical representation is one-hot.  The route identifier is
# intentionally omitted because route reliability is already represented by
# point-in-time numerical priors and one-hotting thousands of routes makes the
# efficient ensemble unnecessarily memory intensive.  This choice is part of the
# model specification and is reported, rather than being a hidden preprocessing
# side effect.
TABM_CATEGORICAL_FEATURES = (
    "Reporting_Airline",
    "Origin",
    "Dest",
    "DistanceBand",
)


@dataclass(slots=True)
class ChimeraBoostModel:
    """A serializable ChimeraBoost estimator with frozen categorical coding."""

    estimator: Any
    codec: CategoryCodec
    task: TaskName

    def predict_proba(self, frame: pd.DataFrame) -> NDArray[np.float64]:
        matrix = self.codec.transform(frame)
        return np.asarray(self.estimator.predict_proba(matrix), dtype=np.float64)


def fit_chimeraboost(
    train_frame: pd.DataFrame,
    train_labels: NDArray[np.int64],
    *,
    task: TaskName,
    params: dict[str, Any],
    validation_frame: pd.DataFrame | None = None,
    validation_labels: NDArray[np.int64] | None = None,
    sample_weight: NDArray[np.float64] | None = None,
) -> ChimeraBoostModel:
    """Fit ChimeraBoost without allowing its preprocessing to see evaluation rows."""

    from chimeraboost import ChimeraBoostClassifier

    codec = CategoryCodec.fit(train_frame)
    train_matrix = codec.transform(train_frame)
    defaults: dict[str, Any] = {
        "quality": 3,
        "random_state": 20260903,
        "thread_count": -1,
        "verbose": False,
    }
    defaults.update(params)
    estimator = ChimeraBoostClassifier(**defaults)
    categorical_indices = [train_matrix.columns.get_loc(name) for name in CATEGORICAL_FEATURES]
    fit_kwargs: dict[str, Any] = {
        "X": train_matrix,
        "y": train_labels,
        "cat_features": categorical_indices,
        "sample_weight": sample_weight,
    }
    if validation_frame is not None and validation_labels is not None:
        fit_kwargs["eval_set"] = (codec.transform(validation_frame), validation_labels)
    else:
        # Holding whole dates together prevents the automatic early-stopping split
        # from training on one flight and validating on another flight from the same
        # operating day.
        fit_kwargs["groups"] = train_frame["FlightDate"].astype(str).to_numpy()
    estimator.fit(**fit_kwargs)
    return ChimeraBoostModel(estimator=estimator, codec=codec, task=task)


@dataclass(slots=True)
class TabMPreprocessor:
    """Frozen train-only preprocessing for TabM."""

    numeric_features: tuple[str, ...]
    categorical_features: tuple[str, ...]
    categorical_levels: dict[str, tuple[str, ...]]
    quantile_transformer: Any
    matrix_feature_columns: tuple[str, ...] | None = None
    numeric_fill_values: dict[str, float] | None = None

    @classmethod
    def fit(
        cls,
        frame: pd.DataFrame,
        *,
        seed: int,
        quantile_subsample: int,
        categorical_features: tuple[str, ...] = TABM_CATEGORICAL_FEATURES,
        feature_columns: tuple[str, ...] | None = None,
    ) -> TabMPreprocessor:
        matrix = (
            engineer_model_features(frame)
            if feature_columns is None
            else frame.loc[:, list(feature_columns)].copy()
        )
        if not set(categorical_features).issubset(matrix.columns):
            raise ValueError("TabM categorical columns are absent from the declared matrix")
        numeric_candidates = (
            NUMERIC_MODEL_FEATURES if feature_columns is None
            else tuple(name for name in feature_columns if name not in categorical_features)
        )
        numeric_features = tuple(
            name
            for name in numeric_candidates
            if matrix[name].nunique(dropna=False) > 1
        )
        if not numeric_features:
            raise ValueError("TabM requires at least one nonconstant numerical feature")
        fill_values = None
        if feature_columns is not None:
            fill_values = {
                name: float(matrix[name].median()) if matrix[name].notna().any() else 0.0
                for name in numeric_features
            }
            matrix[list(numeric_features)] = matrix[list(numeric_features)].fillna(fill_values)
        numeric = matrix.loc[:, list(numeric_features)].to_numpy(dtype=np.float32)
        if not np.isfinite(numeric).all():
            raise ValueError("TabM numerical training inputs contain non-finite values")
        n_quantiles = max(1, min(1_000, len(numeric)))
        transformer = QuantileTransformer(
            n_quantiles=n_quantiles,
            output_distribution="normal",
            subsample=max(n_quantiles, min(quantile_subsample, len(numeric))),
            random_state=seed,
            copy=True,
        ).fit(numeric)
        levels = {
            name: tuple(
                sorted(matrix[name].astype("string").fillna("__MISSING__").unique())
            )
            for name in categorical_features
        }
        return cls(
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            categorical_levels=levels,
            quantile_transformer=transformer,
            matrix_feature_columns=feature_columns,
            numeric_fill_values=fill_values,
        )

    @property
    def categorical_cardinalities(self) -> list[int]:
        # Code zero is reserved for categories not present in training.
        return [len(self.categorical_levels[name]) + 1 for name in self.categorical_features]

    def transform(
        self,
        frame: pd.DataFrame,
    ) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
        columns = getattr(self, "matrix_feature_columns", None)
        matrix = engineer_model_features(frame) if columns is None else frame.loc[:, list(columns)].copy()
        fill_values = getattr(self, "numeric_fill_values", None)
        if fill_values is not None:
            matrix[list(self.numeric_features)] = matrix[list(self.numeric_features)].fillna(fill_values)
        numeric_raw = matrix.loc[:, list(self.numeric_features)].to_numpy(dtype=np.float32)
        if not np.isfinite(numeric_raw).all():
            raise ValueError("TabM numerical inputs contain non-finite values")
        numeric = np.asarray(
            self.quantile_transformer.transform(numeric_raw),
            dtype=np.float32,
        )
        categorical = np.empty(
            (len(matrix), len(self.categorical_features)),
            dtype=np.int64,
        )
        for index, name in enumerate(self.categorical_features):
            encoded = pd.Categorical(
                matrix[name].astype("string").fillna("__MISSING__"),
                categories=list(self.categorical_levels[name]),
            ).codes
            categorical[:, index] = np.asarray(encoded, dtype=np.int64) + 1
        return np.ascontiguousarray(numeric), np.ascontiguousarray(categorical)


def _quantile_bin_edges(
    numeric: NDArray[np.float32],
    *,
    n_bins: int,
    sample_rows: int,
    seed: int,
) -> tuple[NDArray[np.float32], ...]:
    if n_bins <= 1:
        raise ValueError("n_bins must exceed one")
    if len(numeric) > sample_rows:
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(numeric), size=sample_rows, replace=False)
        source = numeric[indices]
    else:
        source = numeric
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges: list[NDArray[np.float32]] = []
    for column in range(source.shape[1]):
        values = np.unique(np.quantile(source[:, column], quantiles)).astype(np.float32)
        if len(values) < 2:
            raise ValueError(f"numerical feature {column} became constant while computing bins")
        edges.append(values)
    return tuple(edges)


def _build_tabm_module(
    *,
    n_num_features: int,
    cat_cardinalities: list[int],
    n_classes: int,
    bin_edges: tuple[NDArray[np.float32], ...],
    model_params: dict[str, Any],
    device: Any,
) -> Any:
    import rtdl_num_embeddings
    import tabm
    import torch

    embedding_kind = str(model_params["embedding"])
    if embedding_kind == "piecewise":
        bins = [torch.as_tensor(values, device=device) for values in bin_edges]
        with warnings.catch_warnings():
            # Binary indicators legitimately have one interval, for which the
            # implementation reduces to min-max scaling.  The package warns on
            # every reconstruction even though this is an expected, recorded case.
            warnings.filterwarnings(
                "ignore",
                message=r"The .* feature has just two bin edges.*",
                category=UserWarning,
            )
            embeddings: Any = rtdl_num_embeddings.PiecewiseLinearEmbeddings(
                bins,
                d_embedding=int(model_params["d_embedding"]),
                activation=False,
                version="B",
            )
    elif embedding_kind == "periodic":
        embeddings = rtdl_num_embeddings.PeriodicEmbeddings(
            n_num_features,
            d_embedding=int(model_params["d_embedding"]),
            n_frequencies=int(model_params["n_frequencies"]),
            frequency_init_scale=float(model_params["frequency_init_scale"]),
            activation=True,
            lite=True,
        )
    elif embedding_kind == "linear_relu":
        embeddings = rtdl_num_embeddings.LinearReLUEmbeddings(
            n_num_features,
            d_embedding=int(model_params["d_embedding"]),
        )
    elif embedding_kind == "none":
        embeddings = None
    else:
        raise ValueError(f"unsupported TabM numerical embedding: {embedding_kind}")

    return tabm.TabM.make(
        n_num_features=n_num_features,
        cat_cardinalities=cat_cardinalities,
        d_out=n_classes,
        num_embeddings=embeddings,
        arch_type=str(model_params["arch_type"]),
        k=int(model_params["k"]),
        n_blocks=int(model_params["n_blocks"]),
        d_block=int(model_params["d_block"]),
        dropout=float(model_params["dropout"]),
    ).to(device)


def _tabm_probabilities(
    model: Any,
    numeric: Any,
    categorical: Any,
    *,
    batch_size: int,
    amp_enabled: bool,
    device_type: str,
) -> NDArray[np.float64]:
    import torch

    model.eval()
    outputs: list[NDArray[np.float32]] = []
    with torch.inference_mode():
        for start in range(0, len(numeric), batch_size):
            stop = min(start + batch_size, len(numeric))
            with torch.autocast(
                device_type=device_type,
                dtype=torch.bfloat16,
                enabled=amp_enabled,
            ):
                logits = model(numeric[start:stop], categorical[start:stop])
            # The TabM heads form an ensemble.  For classification, probabilities
            # (not logits) are averaged as prescribed by the reference implementation.
            probabilities = torch.softmax(logits.float(), dim=-1).mean(dim=1)
            outputs.append(probabilities.cpu().numpy().astype(np.float32, copy=False))
    combined = np.asarray(np.concatenate(outputs, axis=0), dtype=np.float64)
    # The float32 transfer from GPU can perturb a row sum by a few ulps.  Exact
    # renormalisation avoids downstream proper-scoring implementations treating
    # otherwise valid softmax outputs as malformed probabilities.
    return np.asarray(combined / combined.sum(axis=1, keepdims=True), dtype=np.float64)


@dataclass(slots=True)
class TabMModel:
    """Serializable fitted TabM state with lazy device reconstruction."""

    preprocessor: TabMPreprocessor
    task: TaskName
    model_params: dict[str, Any]
    bin_edges: tuple[NDArray[np.float32], ...]
    state_dict: dict[str, Any]
    n_classes: int
    best_epoch: int
    training_history: list[dict[str, int | float | None]]
    fit_device: str

    def predict_proba(self, frame: pd.DataFrame) -> NDArray[np.float64]:
        import torch

        numeric_numpy, categorical_numpy = self.preprocessor.transform(frame)
        requested_device = str(self.model_params.get("inference_device", "auto"))
        if requested_device == "auto":
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(requested_device)
        model = _build_tabm_module(
            n_num_features=numeric_numpy.shape[1],
            cat_cardinalities=self.preprocessor.categorical_cardinalities,
            n_classes=self.n_classes,
            bin_edges=self.bin_edges,
            model_params=self.model_params,
            device=device,
        )
        model.load_state_dict(self.state_dict)
        numeric = torch.as_tensor(numeric_numpy, device=device)
        categorical = torch.as_tensor(categorical_numpy, device=device)
        result = _tabm_probabilities(
            model,
            numeric,
            categorical,
            batch_size=int(self.model_params["eval_batch_size"]),
            amp_enabled=bool(self.model_params["amp"]) and device.type == "cuda",
            device_type=device.type,
        )
        del model, numeric, categorical, numeric_numpy, categorical_numpy
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        return result


def fit_tabm(
    train_frame: pd.DataFrame,
    train_labels: NDArray[np.int64],
    *,
    task: TaskName,
    params: dict[str, Any],
    validation_frame: pd.DataFrame | None = None,
    validation_labels: NDArray[np.int64] | None = None,
    sample_weight: NDArray[np.float64] | None = None,
) -> TabMModel:
    """Fit TabM with independent head losses and probability-space inference."""

    import torch
    import torch.nn.functional as functional

    defaults: dict[str, Any] = {
        "seed": 20260903,
        "device": "auto",
        "inference_device": "auto",
        "batch_size": 1_024,
        "eval_batch_size": 4_096,
        "max_epochs": 40,
        "patience": 7,
        "min_delta": 1e-5,
        "learning_rate": 2e-3,
        "weight_decay": 3e-4,
        "gradient_clipping_norm": 1.0,
        "quantile_subsample": 200_000,
        "bin_sample_rows": 200_000,
        "n_bins": 48,
        "embedding": "piecewise",
        "d_embedding": 8,
        "n_frequencies": 24,
        "frequency_init_scale": 0.01,
        "arch_type": "tabm",
        "k": 16,
        "n_blocks": 2,
        "d_block": 256,
        "dropout": 0.1,
        "amp": True,
        "verbose": False,
        "include_route": False,
    }
    defaults.update(params)
    seed = int(defaults["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    requested_device = str(defaults["device"])
    if requested_device == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(requested_device)
    amp_enabled = bool(defaults["amp"]) and device.type == "cuda"

    preprocessor = TabMPreprocessor.fit(
        train_frame,
        seed=seed,
        quantile_subsample=int(defaults["quantile_subsample"]),
        categorical_features=tuple(defaults.get("categorical_features", (
            CATEGORICAL_FEATURES if bool(defaults["include_route"]) else TABM_CATEGORICAL_FEATURES
        ))),
        feature_columns=(
            tuple(defaults["feature_columns"]) if defaults.get("feature_columns") is not None else None
        ),
    )
    train_numeric_numpy, train_categorical_numpy = preprocessor.transform(train_frame)
    bin_edges = _quantile_bin_edges(
        train_numeric_numpy,
        n_bins=int(defaults["n_bins"]),
        sample_rows=int(defaults["bin_sample_rows"]),
        seed=seed,
    )
    train_numeric = torch.as_tensor(train_numeric_numpy, device=device)
    train_categorical = torch.as_tensor(train_categorical_numpy, device=device)
    labels = torch.as_tensor(train_labels, dtype=torch.long, device=device)
    weights = (
        None
        if sample_weight is None
        else torch.as_tensor(sample_weight, dtype=torch.float32, device=device)
    )
    if len(labels) != len(train_numeric):
        raise ValueError("TabM labels and training rows are misaligned")

    validation_numeric = validation_categorical = validation_labels_array = None
    if validation_frame is not None or validation_labels is not None:
        if validation_frame is None or validation_labels is None:
            raise ValueError("TabM validation frame and labels must be supplied together")
        validation_numeric_numpy, validation_categorical_numpy = preprocessor.transform(
            validation_frame
        )
        validation_numeric = torch.as_tensor(validation_numeric_numpy, device=device)
        validation_categorical = torch.as_tensor(validation_categorical_numpy, device=device)
        validation_labels_array = np.asarray(validation_labels, dtype=np.int64)

    n_classes = 3 if task == "joint" else 2
    model = _build_tabm_module(
        n_num_features=train_numeric.shape[1],
        cat_cardinalities=preprocessor.categorical_cardinalities,
        n_classes=n_classes,
        bin_edges=bin_edges,
        model_params=defaults,
        device=device,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(defaults["learning_rate"]),
        weight_decay=float(defaults["weight_decay"]),
    )
    generator = torch.Generator(device=device).manual_seed(seed)
    batch_size = int(defaults["batch_size"])
    max_epochs = int(defaults["max_epochs"])
    patience = int(defaults["patience"])
    if batch_size <= 0 or max_epochs <= 0 or patience < 0:
        raise ValueError("TabM batch size/epochs must be positive and patience non-negative")

    best_loss = math.inf
    best_epoch = 0
    best_state: dict[str, Any] | None = None
    epochs_without_improvement = 0
    history: list[dict[str, int | float | None]] = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        order: Any = torch.randperm(len(labels), generator=generator, device=device)
        running_loss = 0.0
        running_rows = 0
        for indices in order.split(batch_size):
            optimizer.zero_grad(set_to_none=True)
            loss: Any
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=amp_enabled,
            ):
                logits = model(train_numeric[indices], train_categorical[indices])
                head_count = logits.shape[1]
                per_head_loss = functional.cross_entropy(
                    logits.flatten(0, 1),
                    labels[indices].repeat_interleave(head_count),
                    reduction="none",
                ).reshape(len(indices), head_count)
                if weights is None:
                    loss = per_head_loss.mean()
                else:
                    batch_weights = weights[indices]
                    loss = (per_head_loss * batch_weights[:, None]).sum() / (
                        batch_weights.sum() * head_count
                    )
            loss.backward()
            gradient_norm = float(defaults["gradient_clipping_norm"])
            if gradient_norm > 0:
                torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), gradient_norm)
            optimizer.step()
            running_loss += float(loss.detach().cpu()) * len(indices)
            running_rows += len(indices)

        validation_loss: float | None = None
        if (
            validation_numeric is not None
            and validation_categorical is not None
            and validation_labels_array is not None
        ):
            probabilities = _tabm_probabilities(
                model,
                validation_numeric,
                validation_categorical,
                batch_size=int(defaults["eval_batch_size"]),
                amp_enabled=amp_enabled,
                device_type=device.type,
            )
            validation_loss = float(
                log_loss(validation_labels_array, probabilities, labels=list(range(n_classes)))
            )
            del probabilities
            comparison_loss = validation_loss
        else:
            comparison_loss = running_loss / running_rows

        history.append(
            {
                "epoch": epoch,
                "train_loss": running_loss / running_rows,
                "validation_log_loss": validation_loss,
            }
        )
        if bool(defaults["verbose"]):
            validation_text = (
                "none" if validation_loss is None else f"{validation_loss:.8f}"
            )
            print(
                f"TabM epoch={epoch} train_loss={running_loss / running_rows:.8f} "
                f"validation_log_loss={validation_text}",
                flush=True,
            )
        if comparison_loss < best_loss - float(defaults["min_delta"]):
            best_loss = comparison_loss
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if validation_loss is not None and epochs_without_improvement > patience:
            break

    if best_state is None:
        raise RuntimeError("TabM training completed without a finite checkpoint")
    del model, optimizer, train_numeric, train_categorical, labels, weights
    del train_numeric_numpy, train_categorical_numpy
    if validation_numeric is not None:
        del validation_numeric, validation_categorical
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return TabMModel(
        preprocessor=preprocessor,
        task=task,
        model_params=defaults,
        bin_edges=bin_edges,
        state_dict=best_state,
        n_classes=n_classes,
        best_epoch=best_epoch,
        training_history=history,
        fit_device=str(device),
    )


def frontier_model_features(
    tabm_categorical_features: tuple[str, ...] = TABM_CATEGORICAL_FEATURES,
) -> dict[str, list[str]]:
    """Return the auditable feature profile for frontier estimators."""

    return {
        "chimeraboost": list(MODEL_FEATURES),
        "tabm_categorical": list(tabm_categorical_features),
        "tabm_numerical": list(NUMERIC_MODEL_FEATURES),
        "tabm_excluded_categorical": sorted(
            set(CATEGORICAL_FEATURES) - set(tabm_categorical_features)
        ),
    }
