import logging
import os
from functools import lru_cache
from typing import Any, List, Optional, Tuple, Union

import click
import numpy as np
import rich
import torch
from PIL import Image
from rich.progress import track
from rich.table import Table

from vdtk.data_utils import Sample, b64_to_bin, load_dataset
from vdtk.score import _handle_baseline_index
from vdtk.third_party.clip import clip
from vdtk.utils.rich import baseline_column

Result = List[
    Tuple[
        Tuple[List[np.floating], List[np.floating], List[np.floating], List[np.floating], List[np.floating]],
        Tuple[List[np.floating], List[np.floating], List[np.floating], List[np.floating], List[np.floating]],
    ],
]


@lru_cache
def clip_model() -> Tuple[Any, Any, Any]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess, device


@lru_cache
def _get_feature(media: Union[str, bytes]) -> torch.Tensor:
    model, preprocess, device = clip_model()
    image = preprocess(Image.open(media)).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    return image_features.reshape(-1)


def _get_image_feature_db(data: List[Sample]) -> torch.Tensor:
    features = []
    for sample in track(data, description="Featurizing dataset", transient=True):
        # FIXME: If media is both b64 and path, only the b64 is used
        # we should raise an error or warning if both are present
        if sample.media_b64 is not None:
            features.append(_get_feature(b64_to_bin(sample.media_b64)))
        elif sample.media_path is not None:
            features.append(_get_feature(sample.media_path))
    return torch.stack(features).to("cpu" if not torch.cuda.is_available() else "cuda")


def _get_text_features(
    sample: Sample, text_features: torch.Tensor, char_limit: int = 300
) -> Tuple[torch.Tensor, torch.Tensor]:
    model, _, device = clip_model()
    candidate_text = clip.tokenize([i[:char_limit] for i in sample.candidates]).to(device)
    reference_text = clip.tokenize([i[:char_limit] for i in sample.references]).to(device)
    with torch.no_grad():
        candidate_text_features = model.encode_text(candidate_text)
        reference_text_features = model.encode_text(reference_text)
        candidate_text_features /= candidate_text_features.norm(dim=-1, keepdim=True)
        reference_text_features /= reference_text_features.norm(dim=-1, keepdim=True)

    return candidate_text_features, reference_text_features


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, data, preprocess):
        self.data = data
        self.preprocess = preprocess

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        if sample.media_b64 is not None:
            media = b64_to_bin(sample.media_b64)
        elif sample.media_path is not None:
            media = sample.media_path
        return self.preprocess(Image.open(media))


def _get_image_feature_db_by_batch(data: List[Sample], batch_size, num_workers) -> torch.Tensor:
    model, preprocess, device = clip_model()
    dataloader = torch.utils.data.DataLoader(
        ImageDataset(data, preprocess), batch_size=batch_size, num_workers=num_workers
    )

    features = []
    for batch in track(dataloader, description="Featurizing dataset", transient=True):
        with torch.no_grad():
            image_features = model.encode_image(batch.to(device))
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        features.append(image_features)

    features = torch.cat(features)
    return features


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, data, char_limit: int = 300):
        self.data = data
        self.char_limit = char_limit

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        candidate_tokens = clip.tokenize([i[: self.char_limit] for i in sample.candidates])
        reference_toekns = clip.tokenize([i[: self.char_limit] for i in sample.references])
        candidate_count = len(sample.candidates)
        reference_count = len(sample.references)
        return candidate_tokens, reference_toekns, candidate_count, reference_count, idx

    def collate_fn(self, samples):
        candidate_tokens, reference_toekns, candidate_count, reference_count, batch_idx = zip(*samples)
        candidate_tokens = torch.cat(candidate_tokens)
        reference_toekns = torch.cat(reference_toekns)
        batch_idx = torch.tensor(batch_idx)
        return candidate_tokens, reference_toekns, candidate_count, reference_count, batch_idx


def _add_table_row(
    i: int,
    baseline_index: Optional[int],
    table: Table,
    name: str,
    scores: np.ndarray,
    outputs: Result,
    is_candidate: bool,
) -> None:
    (rank, rrank, recall_1, recall_5, recall_max, similarity) = scores

    if i is None or baseline_index is None:
        table.add_row(
            name,
            f"{np.mean(rank):.4f} ± {np.std(rank):.4f}",
            f"{np.mean(rrank):.4f} ± {np.std(rrank):.4f}",
            f"{np.mean(recall_1):.4f} ± {np.std(recall_1):.4f}",
            f"{np.mean(recall_5):.4f} ± {np.std(recall_5):.4f}",
            f"{np.amax(recall_max):.4f}",
            f"{np.mean(similarity):.4f} ± {np.std(similarity):.4f}",
        )
    else:
        if i == baseline_index and not is_candidate:
            table.add_row(
                name,
                f"{np.mean(rank):.4f} ± {np.std(rank):.4f}",
                f"{np.mean(rrank):.4f} ± {np.std(rrank):.4f}",
                f"{np.mean(recall_1):.4f} ± {np.std(recall_1):.4f}",
                f"{np.mean(recall_5):.4f} ± {np.std(recall_5):.4f}",
                f"{np.amax(recall_max):.4f}",
                f"{np.mean(similarity):.4f} ± {np.std(similarity):.4f}",
            )
        else:
            table.add_row(
                name,
                baseline_column(rank, outputs[baseline_index][1][0], positive=False),  # type: ignore
                baseline_column(rrank, outputs[baseline_index][1][1]),  # type: ignore
                baseline_column(recall_1, outputs[baseline_index][1][2]),  # type: ignore
                baseline_column(recall_5, outputs[baseline_index][1][3]),  # type: ignore
                baseline_column(
                    recall_max,
                    outputs[baseline_index][1][4],  # type: ignore
                    aggregate=np.amax,
                    baseline_aggregate=np.amax,
                    positive=False,
                ),
                baseline_column(similarity, outputs[baseline_index][1][5], positive=False),  # type: ignores
            )


@click.command()
@click.argument("dataset_paths", type=str, nargs=-1)
@click.option("--split", default=None, type=str, help="Split to evaluate")
@click.option("--media-root", default=None, type=str, help="Root directory for media")
@click.option("--batch-size", default=512, type=int, help="Batch size for featurization")
@click.option("--num-workers", default=8, type=int, help="Number of processes to use")
@click.option("--debug", is_flag=True, help="Debug mode")
@click.option("--export-scores", is_flag=True, help="Export scores to a csv file")
def clip_recall(
    dataset_paths: List[str],
    split: Optional[str] = None,
    media_root: Optional[str] = None,
    batch_size: int = 512,
    num_workers: int = 8,
    debug: bool = False,
    export_scores: bool = False,
) -> None:
    # Get the baseline
    baseline_index, dataset_paths = _handle_baseline_index(dataset_paths)

    outputs: List[Result] = []
    for ds in dataset_paths:
        data = load_dataset(ds, media_root)
        if split is not None:
            # Filter the data for the correct split
            data = [s for s in data if s.split == split]

        if len(data) == 0:
            logging.error(f"Dataset {ds} has no samples")
            continue

        if debug:
            data = data[:1000]

        # Compute the features
        image_feature_db = _get_image_feature_db_by_batch(data, batch_size, num_workers)

        # Compute the recall
        candidate_scores = []
        candidate_similarity_scores = []
        reference_similarity_scores = []
        reference_scores = []
        text_dataset = TextDataset(data)
        text_dataloader = torch.utils.data.DataLoader(
            text_dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=text_dataset.collate_fn
        )

        model, _, device = clip_model()
        for batch in track(
            text_dataloader, description=f"Computing recall for dataset {os.path.basename(ds)}", transient=True
        ):
            condidate_tokens, reference_tokens, candidate_count, reference_count, batch_idx = batch
            condidate_tokens = condidate_tokens.to(device, non_blocking=True)
            reference_tokens = reference_tokens.to(device, non_blocking=True)
            with torch.no_grad():
                candidate_features = model.encode_text(condidate_tokens)
                reference_features = model.encode_text(reference_tokens)
                candidate_features /= candidate_features.norm(dim=-1, keepdim=True)
                reference_features /= reference_features.norm(dim=-1, keepdim=True)

            # candidate_features, reference_features = _get_text_features(sample, image_feature_db)
            _candidate_similarity_scores = image_feature_db @ candidate_features.T  # num_images x num_candidates
            _candidate_similarity_scores_splits = _candidate_similarity_scores.split(candidate_count, -1)
            # candidate_ranks = (_candidate_similarity_scores > _candidate_similarity_scores[index]).sum(dim=0)
            # Bugggy, only suit for batch_size=1

            _reference_similarity_scores = image_feature_db @ reference_features.T
            _reference_similarity_scores_splits = _reference_similarity_scores.split(reference_count, -1)
            # reference_ranks = (_reference_similarity_scores > _reference_similarity_scores[index]).sum(dim=0)
            # Bugggy, only suit for batch_size=1

            # Bugggy, only suit for batch_size=1
            # candidate_scores.extend([i.cpu().numpy() for i in (candidate_ranks + 1).split(candidate_count, -1)])
            # reference_scores.extend([i.cpu().numpy() for i in (reference_ranks + 1).split(reference_count, -1)])
            for col_id, (_candidate_similarity_scores_split, _reference_similarity_scores_split) in enumerate(
                zip(_candidate_similarity_scores_splits, _reference_similarity_scores_splits)
            ):
                row_id = batch_idx[col_id]
                _candindate_ranks = (
                    _candidate_similarity_scores_split > _candidate_similarity_scores_split[row_id]
                ).sum(0)
                candidate_scores.append(_candindate_ranks.cpu().numpy() + 1)
                _reference_ranks = (
                    _reference_similarity_scores_split > _reference_similarity_scores_split[row_id]
                ).sum(0)
                reference_scores.append(_reference_ranks.cpu().numpy() + 1)

            for col_id, row_id in enumerate(batch_idx):
                candidate_similarity_scores.append(_candidate_similarity_scores[row_id, col_id].cpu().numpy())
                reference_similarity_scores.append(_reference_similarity_scores[row_id, col_id].cpu().numpy())

        outputs.append(
            (
                (
                    # rank
                    [np.mean(i) for i in candidate_scores],
                    # Reciprocal rank
                    [np.mean(1 / i) for i in candidate_scores],
                    # Recall at 1
                    [np.mean(i <= 1) for i in candidate_scores],
                    # Recall at 5
                    [np.mean(i <= 5) for i in candidate_scores],
                    # 100% recall at
                    [np.amax(i) for i in candidate_scores],  # type: ignore
                    # Similarity scores
                    candidate_similarity_scores,
                ),
                (
                    # rank
                    [np.mean(i) for i in reference_scores],
                    # Reciprocal rank
                    [np.mean(1 / i) for i in reference_scores],
                    # Recall at 1
                    [np.mean(i <= 1) for i in reference_scores],
                    # Recall at 5
                    [np.mean(i <= 5) for i in reference_scores],
                    # 100% recall at
                    [np.amax(i) for i in reference_scores],
                    # Similarity scores
                    reference_similarity_scores,
                ),
            )
        )
    if export_scores:
        for dataset_path, output in zip(dataset_paths, outputs):
            output = np.array(output)
            output = output.reshape(-1, output.shape[-1])
            np.savetxt(
                os.path.join(os.path.dirname(dataset_path), f"{os.path.basename(dataset_path)}_clip_recall.csv"),
                output.T,
                delimiter=",",
                header=",".join(
                    [
                        "candidate_rank",
                        "candidate_rrank",
                        "candidate_recall_1",
                        "candidate_recall_5",
                        "candidate_recall_max",
                        "candidate_similarity",
                        "reference_rank",
                        "reference_rrank",
                        "reference_recall_1",
                        "reference_recall_5",
                        "reference_recall_max",
                        "reference_similarity",
                    ]
                ),
            )

    # Print the results
    table = Table(title=f"CLIP Recall")
    table.add_column("Dataset", justify="left", style="cyan", no_wrap=True)
    table.add_column("Mean Rank", justify="right", style="magenta")
    table.add_column("Mean Reciprocal Rank", justify="right", style="magenta")
    table.add_column("Recall @ 1", justify="right", style="magenta")
    table.add_column("Recall @ 5", justify="right", style="magenta")
    table.add_column("100% Recall", justify="right", style="magenta")
    table.add_column("CLIP similarity", justify="right", style="magenta")
    for col_id, (ds, (candidate_scores, reference_scores)) in enumerate(zip(dataset_paths, outputs)):  # type: ignore
        # Add The candidate scores
        _add_table_row(
            col_id,
            baseline_index,
            table,
            os.path.basename(ds) + f" (candidate) #{len(candidate_scores[0])}",
            candidate_scores,  # type: ignore
            outputs,  # type: ignore
            True,
        )
        _add_table_row(
            col_id,
            baseline_index,
            table,
            os.path.basename(ds) + f" (reference) #{len(reference_scores[0])}",
            reference_scores,  # type: ignore
            outputs,  # type: ignore
            False,
        )
    rich.print(table)
