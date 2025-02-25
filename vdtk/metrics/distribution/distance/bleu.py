# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Python implementation of BLEU and smooth-BLEU.
This module provides a Python implementation of BLEU and smooth-BLEU.
Smooth BLEU is computed following the method outlined in the paper:
Chin-Yew Lin, Franz Josef Och. ORANGE: a method for evaluating automatic
evaluation metrics for machine translation. COLING 2004.
"""

import collections
import math
from functools import lru_cache
from typing import Counter, List, Sequence, Tuple

from . import DistanceFunction
from .normalize import coco_normalize


def _get_ngrams(segment: Sequence[str], max_order: int) -> Counter[Tuple[str, ...]]:
    """Extracts all n-grams upto a given maximum order from an input segment.
    Args:
      segment: text segment from which n-grams will be extracted.
      max_order: maximum length in tokens of the n-grams returned by this
          methods.
    Returns:
      The Counter containing all n-grams upto max_order in segment
      with a count of how many times each n-gram occurred.
    """
    ngram_counts: Counter[Tuple[str, ...]] = collections.Counter()
    for order in range(1, max_order + 1):
        for i in range(len(segment) - order + 1):
            ngram = tuple(segment[i : i + order])
            ngram_counts[ngram] += 1
    return ngram_counts


def _compute_bleu(
    reference_corpus: Sequence[Sequence[Sequence[str]]],
    translation_corpus: Sequence[Sequence[str]],
    max_order: int = 4,
    smooth: bool = False,
) -> Tuple[float, List[float], float, float, int, int]:
    """Computes BLEU score of translated segments against one or more references.
    Args:
      reference_corpus: list of lists of references for each translation. Each
          reference should be tokenized into a list of tokens.
      translation_corpus: list of translations to score. Each translation
          should be tokenized into a list of tokens.
      max_order: Maximum n-gram order to use when computing BLEU score.
      smooth: Whether or not to apply Lin et al. 2004 smoothing.
    Returns:
      3-Tuple with the BLEU score, n-gram precisions, geometric mean of n-gram
      precisions and brevity penalty.
    """
    matches_by_order = [0] * max_order
    possible_matches_by_order = [0] * max_order
    reference_length = 0
    translation_length = 0
    for (references, translation) in zip(reference_corpus, translation_corpus):
        reference_length += min(len(r) for r in references)
        translation_length += len(translation)

        merged_ref_ngram_counts: Counter[Tuple[str, ...]] = collections.Counter()
        for reference in references:
            merged_ref_ngram_counts |= _get_ngrams(reference, max_order)
        translation_ngram_counts = _get_ngrams(translation, max_order)
        overlap = translation_ngram_counts & merged_ref_ngram_counts
        for ngram in overlap:
            matches_by_order[len(ngram) - 1] += overlap[ngram]
        for order in range(1, max_order + 1):
            possible_matches = len(translation) - order + 1
            if possible_matches > 0:
                possible_matches_by_order[order - 1] += possible_matches

    precisions = [0.0] * max_order
    for i in range(max_order):
        if smooth:
            precisions[i] = (matches_by_order[i] + 1.0) / (possible_matches_by_order[i] + 1.0)
        elif possible_matches_by_order[i] > 0:
            precisions[i] = float(matches_by_order[i]) / possible_matches_by_order[i]
        else:
            precisions[i] = 0.0

    if min(precisions) > 0:
        p_log_sum = sum((1.0 / max_order) * math.log(p) for p in precisions)
        geo_mean = math.exp(p_log_sum)
    else:
        geo_mean = 0

    ratio = float(translation_length) / (reference_length + 1e-9)

    bp = 1.0 if ratio > 1.0 else math.exp(1 - 1.0 / (ratio + 1e-9))
    bleu = geo_mean * bp

    return (bleu, precisions, bp, ratio, translation_length, reference_length)


def compute_bleu(samples: Sequence[str], references: Sequence[str]) -> float:
    return _compute_bleu(
        [[coco_normalize(r).split()] for r in references], [coco_normalize(s).split() for s in samples]
    )[0]


def compute_bleu_single(sample: str, references: Sequence[str]) -> float:
    return _compute_bleu([[coco_normalize(r).split() for r in references]], [coco_normalize(sample).split()])[0]


class BLEU4Distance(DistanceFunction):
    @lru_cache(None)
    def __call__(self, x: str, y: str) -> float:
        return 2 - compute_bleu_single(x, [y])
