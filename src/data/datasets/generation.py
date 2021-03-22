import random
from typing import Iterator, Dict, List, Tuple, Iterable, Callable, Optional

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import IterableDataset
from transformers import PreTrainedTokenizer

from src.utils.commons import add_noise_to_value, chunks, flatten
from src.utils.logging import get_project_logger

logger = get_project_logger(__name__)


class GenerativeDataset:
    @classmethod
    def from_lines(cls, lines: Iterable[str], **kwargs):
        raise NotImplementedError

    @classmethod
    def from_file(cls, path: str, **kwargs):
        raise NotImplementedError


class ParallelDataset(GenerativeDataset, IterableDataset):
    @classmethod
    def from_lines(cls, lines: Iterable[str], **kwargs):
        return cls(lambda: lines, **kwargs)

    @classmethod
    def from_file(cls, path: str, **kwargs):
        def r():
            with open(path) as f:
                for line in f:
                    yield line.strip()

        return cls(r, **kwargs)

    def __init__(
        self,
        iterator_generator: Callable[[], Iterable],
        tokenizer: PreTrainedTokenizer,
        for_inference: bool = False,
        max_tokens_per_batch: int = 1024,
        min_length: int = -1,
        max_length: int = -1,
        truncate: bool = False,
        section_size: int = 50_000,
        drop_last_batch: bool = False,
        prebatch: bool = False,
    ):
        self.iterator_generator = iterator_generator
        self.tokenizer = tokenizer
        self.for_inference = for_inference
        self.max_tokens_per_batch = max_tokens_per_batch
        self.min_length = min_length
        self.max_length = max_length
        self.truncate = truncate
        self.section_size = section_size if prebatch else 1
        self.drop_last_batch = drop_last_batch
        self.prebatch = prebatch

    def _generate_dataset(self):
        def prebatch_ds(ds: List[Tuple[List[int], List[int], str, Optional[str]]]):
            ds = sorted(
                ds, key=lambda x: add_noise_to_value(len(x[0]) + len(x[1]), noise_param=0.1)
            )
            ds = list(chunks(ds, 512))
            random.shuffle(ds)
            return flatten(ds)

        logger.info("Initting dataset")

        discarded_due_to_min_length = 0
        discarded_due_to_max_length = 0

        read_samples = 0
        ds = []

        for line in self.iterator_generator():

            if read_samples % 10_000 == 0:
                logger.info(f"{read_samples} entries added to dataset")

            line = line.strip()
            parts = line.split("\t")
            if self.for_inference:
                source = parts[0]
                target = parts[1] if len(parts) == 2 else None
            else:
                source = parts[0]
                target = parts[1]

            # encode
            text_source, text_target = source, target
            if not self.for_inference:
                sample = self.tokenizer.prepare_seq2seq_batch([source], tgt_texts=[target])
                source = sample["input_ids"][0]
                target = sample["labels"][0]
            else:
                sample = self.tokenizer.prepare_seq2seq_batch([source])
                source = sample["input_ids"][0]
                target = [self.tokenizer.bos_token_id]

            # truncate if requested
            if self.truncate:
                source = source[: self.max_length]
                target = target[: self.max_length]

            # check min length
            if self.min_length != -1 and (
                len(source) < self.min_length
                or (len(target) < self.min_length and not self.for_inference)
            ):
                discarded_due_to_min_length += 1
                if discarded_due_to_min_length % 1_000 == 0:
                    logger.warning(
                        f"{discarded_due_to_min_length} samples have been discarded due to being shorter than minimum length {self.min_length}"
                    )
                continue

            # check max length
            if self.max_length != -1 and (
                len(source) > self.max_length or len(target) > self.max_length
            ):
                discarded_due_to_max_length += 1
                if discarded_due_to_max_length % 1_000 == 0:
                    logger.warning(
                        f"{discarded_due_to_max_length} samples have been discarded due to being longer than maximum length {self.max_length}"
                    )
                continue

            ds.append((source, target, text_source, text_target))
            if len(ds) == self.section_size:
                if self.prebatch:
                    ds = prebatch_ds(ds)
                yield from ds
                ds = []

            read_samples += 1

        if len(ds) > 0:
            if self.prebatch:
                ds = prebatch_ds(ds)
            yield from ds

        if discarded_due_to_min_length > 0:
            logger.warning(
                f"{discarded_due_to_min_length} samples have been discarded due to being shorter than minimum length {self.min_length}"
            )

        if discarded_due_to_max_length > 0:
            logger.warning(
                f"{discarded_due_to_max_length} samples have been discarded due to being longer than maximum length {self.max_length}"
            )

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:

        dataset = self._generate_dataset()

        batch = []
        ct = 0

        for sample in dataset:

            sample_tokens = len(sample[0]) + len(sample[1])

            if (
                max(ct, sample_tokens) * (len(batch) + 1) > self.max_tokens_per_batch
                and len(batch) > 0
            ):
                yield self.prepare_output_batch(batch)
                batch = []
                ct = 0

            batch.append(sample)
            ct = max(ct, sample_tokens)

        # drop last cause might be too short and result in issues (nan if we are using amp)
        if not self.drop_last_batch and len(batch) > 0:
            yield self.prepare_output_batch(batch)

    def prepare_output_batch(
        self, batch: List[Tuple[List[int], List[int], str, Optional[str]]]
    ) -> Dict[str, torch.Tensor]:

        try:
            pad_token_id = self.tokenizer.pad_token_id
        except:
            pad_token_id = 0

        # build source
        source = pad_sequence(
            [torch.tensor(e[0]) for e in batch], batch_first=True, padding_value=pad_token_id
        )
        source_padding_mask = pad_sequence(
            [torch.full((len(e[0]),), fill_value=True) for e in batch],
            batch_first=True,
            padding_value=False,
        )

        # build target
        target = pad_sequence(
            [torch.tensor(e[1]) for e in batch], batch_first=True, padding_value=pad_token_id
        )
        target_padding_mask = pad_sequence(
            [torch.full((len(e[1]),), fill_value=True) for e in batch],
            batch_first=True,
            padding_value=False,
        )

        # return
        return {
            "source": source,
            "source_padding_mask": source_padding_mask,
            "target": target,
            "target_padding_mask": target_padding_mask,
            "text_source": [e[2] for e in batch],
            "text_target": [e[3] for e in batch],
        }
