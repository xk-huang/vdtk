import logging
import multiprocessing as mp
import os
from itertools import chain
from typing import List, Optional, Tuple

import click
import numpy as np
import rich
import spacy
from rich.progress import track
from rich.table import Table

from vdtk.data_utils import load_dataset
from vdtk.score import _handle_baseline_index
from vdtk.utils.nlp import get_or_download_spacy_model
from vdtk.utils.rich import baseline_column


def compute_object_roverlap(
    nlp: spacy.language.Language, query: str, targets: List[str], POS: Tuple[str, ...] = ("NOUN",)
) -> float:
    """Compute the object overlap between a query and a list of targets.

    Args:
        query (str): The query.
        targets (List[str]): The list of targets.

    Returns:
        float: The object overlap.
    """
    query_doc = nlp(query)
    targets_doc = nlp(" ".join(targets))
    query_objects = set([token.text for token in query_doc if token.pos_ in POS])
    targets_objects = set([token.text for token in targets_doc if token.pos_ in POS])
    # Return the recall
    # print(query_objects, targets_objects)
    return len(set(query_objects).intersection(set(targets_objects))) / (len(set(targets_objects)) + 1e-8)


def compute_object_rdistance(
    nlp: spacy.language.Language, query: str, targets: List[str], POS: Tuple[str, ...] = ("NOUN",)
) -> float:
    """Compute the object overlap between a query and a list of targets.

    Args:
        query (str): The query.
        targets (List[str]): The list of targets.

    Returns:
        float: The object overlap.
    """
    query_doc = nlp(query)
    targets_doc = nlp(" ".join(targets))
    query_objects = [token for token in query_doc if token.pos_ in POS]
    targets_objects = [token for token in targets_doc if token.pos_ in POS]

    query_uniq = set()
    targets_uniq = set()

    qos = []
    tos = []
    for token in query_objects:
        if token.text not in query_uniq:
            query_uniq.add(token.text)
            qos.append(token)
    for token in targets_objects:
        if token.text not in targets_uniq:
            targets_uniq.add(token.text)
            tos.append(token)

    metric = []
    for q in tos:
        sims = []
        for t in qos:
            sims.append(q.similarity(t))
        metric.append(max(sims) if sims else 0)

    return sum(metric) / (len(metric) + 1e-8)


def _noun_recall_job(_nlp: spacy.language.Language, data: List) -> List:
    return [
        [
            compute_object_roverlap(
                _nlp,
                c,
                sample.references,
                (
                    "NOUN",
                    "PROPN",
                ),
            )
            for c in sample.candidates
        ]
        for sample in track(data, transient=True, description="Computing content NOUN/PROPN recall...")
    ]


def _verb_recall_job(_nlp: spacy.language.Language, data: List) -> List:
    return [
        [
            compute_object_roverlap(
                _nlp,
                c,
                sample.references,
                ("VERB",),
            )
            for c in sample.candidates
        ]
        for sample in track(data, transient=True, description="Computing content VERB recall...")
    ]


def _noun_distance_job(_nlp: spacy.language.Language, data: List) -> List:
    return [
        [
            compute_object_rdistance(
                _nlp,
                c,
                sample.references,
                (
                    "NOUN",
                    "PROPN",
                ),
            )
            for c in sample.candidates
        ]
        for sample in track(data, transient=True, description="Computing content fuzzy NOUN/PROPN recall...")
    ]


def _verb_distance_job(_nlp: spacy.language.Language, data: List) -> List:
    return [
        [
            compute_object_rdistance(
                _nlp,
                c,
                sample.references,
                ("VERB",),
            )
            for c in sample.candidates
        ]
        for sample in track(data, transient=True, description="Computing content fuzzy VERB recall...")
    ]


@click.command()
@click.argument("dataset_paths", type=str, nargs=-1)
@click.option("--split", default=None, type=str, help="Split to evaluate")
@click.option("--num-workers", default=8, type=int, help="Number of processes to use")
@click.option("--save-dist-plot", default=False, is_flag=True, type=bool, help="Save distribution plot")
@click.option("--save-scores", default=False, is_flag=True, type=bool, help="Save scores")
@click.option("--num-samples", default=None, type=int, help="Use less samples")
def content_recall(
    dataset_paths: List[str],
    split: Optional[str] = None,
    num_workers: int = 8,
    save_dist_plot: bool = False,
    save_scores: bool = False,
    num_samples: Optional[int] = None,
) -> None:
    # Get the baseline
    baseline_index, dataset_paths = _handle_baseline_index(dataset_paths)
    _nlp = get_or_download_spacy_model("en_core_web_lg")

    outputs = []
    # for ds in track(dataset_paths, transient=True, description="Computing content recall..."):
    for ds in dataset_paths:
        data = load_dataset(ds)
        if num_samples is not None:
            data = data[: num_samples]
        if split is not None:
            # Filter the data for the correct split
            data = [s for s in data if s.split == split]

        if len(data) == 0:
            logging.error(f"Dataset {ds} has no samples for split {split}.")
            continue

        num_samples = len(data)
        num_samples_per_process = max(1, num_samples // num_workers)

        def get_dataloader():
            yield from (
                (_nlp, data[i : i + num_samples_per_process]) for i in range(0, num_samples, num_samples_per_process)
            )

        with mp.Pool(num_workers) as p:
            noun_recall = p.starmap(_noun_recall_job, get_dataloader())
            noun_recall = list(chain.from_iterable(noun_recall))

        with mp.Pool(num_workers) as p:
            verb_recall = p.starmap(_verb_recall_job, get_dataloader())
            verb_recall = list(chain.from_iterable(verb_recall))

        with mp.Pool(num_workers) as p:
            noun_distance = p.starmap(_noun_distance_job, get_dataloader())
            noun_distance = list(chain.from_iterable(noun_distance))

        with mp.Pool(num_workers) as p:
            verb_distance = list(p.starmap(_verb_distance_job, get_dataloader()))
            verb_distance = list(chain.from_iterable(verb_distance))

        noun_recall_a = np.array(noun_recall)
        verb_recall_a = np.array(verb_recall)
        noun_distance_a = np.array(noun_distance)
        verb_distance_a = np.array(verb_distance)

        outputs.append(
            (
                noun_recall_a,
                verb_recall_a,
                noun_distance_a,
                verb_distance_a,
            )
        )

    # Print the results
    table = Table(title="Content Recall")
    table.add_column("Dataset", justify="left", style="cyan", no_wrap=True)
    table.add_column("Noun Recall", justify="right", style="magenta")
    table.add_column("Verb Recall", justify="right", style="magenta")
    table.add_column("Noun Recall (Fuzzy)", justify="right", style="magenta")
    table.add_column("Verb Recall (Fuzzy)", justify="right", style="magenta")
    for i, (ds, (nr, vr, nd, vd)) in enumerate(zip(dataset_paths, outputs)):
        nr_mean = np.mean(nr, axis=-1)
        vr_mean = np.mean(vr, axis=-1)
        nd_mean = np.mean(nd, axis=-1)
        vd_mean = np.mean(vd, axis=-1)

        if baseline_index is None:
            table.add_row(
                os.path.basename(ds),
                f"{np.mean(nr_mean):.4f} ± {np.std(nr_mean):.4f}",
                f"{np.mean(vr_mean):.4f} ± {np.std(vr_mean):.4f}",
                f"{np.mean(nd_mean):.4f} ± {np.std(nd_mean):.4f}",
                f"{np.mean(vd_mean):.4f} ± {np.std(vd_mean):.4f}",
            )
        else:
            if i == baseline_index:
                table.add_row(
                    os.path.basename(ds),
                    f"{np.mean(nr_mean):.4f} ± {np.std(nr_mean):.4f}",
                    f"{np.mean(vr_mean):.4f} ± {np.std(vr_mean):.4f}",
                    f"{np.mean(nd_mean):.4f} ± {np.std(nd_mean):.4f}",
                    f"{np.mean(vd_mean):.4f} ± {np.std(vd_mean):.4f}",
                )
            else:
                table.add_row(
                    os.path.basename(ds),
                    baseline_column(nr_mean, np.mean(outputs[baseline_index][0], axis=-1)),
                    baseline_column(vr_mean, np.mean(outputs[baseline_index][1], axis=-1)),
                    baseline_column(nd_mean, np.mean(outputs[baseline_index][2], axis=-1)),
                    baseline_column(vd_mean, np.mean(outputs[baseline_index][3], axis=-1)),
                )
        for label, scores in zip(
            ("noun_recall", "verb_recall", "noun_distance", "verb_distance"),
            (nr, vr, nd, vd),
        ):
            if save_dist_plot:
                _plot_distribution(scores.flatten(), _add_prefix_suffix_to_path(ds, label + "-", ".png"), name=label)
            if save_scores:
                _save_scores(scores.flatten().tolist(), _add_prefix_suffix_to_path(ds, label + "-", ".json"))
    rich.print(table)

def _plot_distribution(scores: List[float], output_path: str, name: str) -> None:
    """
    for i, (path, score) in enumerate(zip(dataset_paths, scores)):
        output_path = path + f".{label}.{i}.png"
        _plot_distribution(score[1], output_path, name=label)

    Args:
        scores (List[float]): _description_
        output_path (str): _description_
        name (str): _description_
    """
    import matplotlib.pyplot as plt

    sorted_scores = np.sort(scores)
    y_values_pdf, bin_edges = np.histogram(sorted_scores, bins=100, density=True)
    bin_width = bin_edges[1] - bin_edges[0]
    y_values_pdf_count = y_values_pdf * bin_width
    y_values_cdf = np.arange(len(sorted_scores)) / float(len(sorted_scores))
    fig, ax1 = plt.subplots()
    ax1.hist(sorted_scores, bins=100, density=True, alpha=0.5, color="blue")
    ax1.set_xlabel(name)
    ax1.set_ylabel("PDF", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")
    ax2 = ax1.twinx()
    ax2.plot(sorted_scores, y_values_cdf, color="red")
    ax2.set_ylabel("CDF", color="red")
    ax2.tick_params(axis="y", labelcolor="red")
    plt.savefig(output_path)
    plt.cla()
    print(f"Saved plot to: {output_path}")
    # num_zeros = scores.count(0)
    # total_scores = len(scores)
    # zero_percent = num_zeros / total_scores * 100
    # print(f"Percentage of scores equal to 0: {zero_percent:.2f}% ({total_scores})")
    # print(f"PDF count: {y_values_pdf_count.sum():.2f}")


def _save_scores(scores: List[float], output_path: str) -> None:
    import json
    with open(output_path, "w") as f:
        json.dump(scores, f)
    print(f"Saved scores to: {output_path}")


def _add_prefix_suffix_to_path(path: str, prefix: str, suffix: str) -> str:
    base_dir, filename = os.path.split(path)
    return os.path.join(base_dir, prefix + filename + suffix)