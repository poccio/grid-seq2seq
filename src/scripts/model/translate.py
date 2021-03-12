import argparse
from typing import List, Iterable, Tuple, Optional

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.datasets.generation import ParallelDataset
from src.pl_modules.utils import load_pl_module_from_checkpoint


def translate(
    module: pl.LightningModule,
    sources: Iterable[str],
    num_sequences: int,
    generation_param_conf_path: str,
    token_batch_size: int = 1024,
    progress_bar: bool = False,
) -> Iterable[Tuple[str, List[str], Optional[str]]]:

    module.enable_generation_mode()
    module.load_generation_params(OmegaConf.load(generation_param_conf_path))

    # todo only works on single gpu
    device = next(module.parameters()).device

    dataset = ParallelDataset.from_lines(
        sources,
        tokenizer=module.tokenizer,
        for_inference=True,
        max_tokens_per_batch=token_batch_size,
        drop_last_batch=False,
    )
    dataloader = DataLoader(dataset, batch_size=None, num_workers=0)

    iterator = dataloader
    if progress_bar:
        iterator = tqdm(iterator, desc="Translating")

    for batch in iterator:

        # translate
        with autocast(enabled=True):
            with torch.no_grad():
                batch_generations = module(
                    **{k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()},
                    num_sequences=num_sequences,
                )
                batch_input = batch["text_source"]
                batch_gold_output = batch["text_target"]
                batch_generations = batch_generations.generation

        # generate
        for sample_input, sample_generations, sample_gold_output in zip(
            batch_input, batch_generations, batch_gold_output
        ):
            decoded_sample_generations = []
            for sample_generation in module.tokenizer.batch_decode(
                sample_generations, clean_up_tokenization_spaces=False
            ):
                if module.tokenizer.eos_token in sample_generation:
                    sample_generation = sample_generation[
                        : sample_generation.index(module.tokenizer.eos_token)
                        + len(module.tokenizer.eos_token)
                    ]
                decoded_sample_generations.append(sample_generation)

            yield sample_input, decoded_sample_generations, sample_gold_output

    module.disable_generation_mode()


def interactive_main(
    model_checkpoint_path: str,
    num_sequences: int,
    generation_param_conf_path: str,
    cuda_device: int,
):

    model = load_pl_module_from_checkpoint(model_checkpoint_path)
    model.to(torch.device(cuda_device if cuda_device != -1 else "cpu"))
    model.eval()
    model.freeze()

    while True:
        source = input("Enter source text: ").strip()
        _, predictions, _ = next(
            translate(
                model,
                [source],
                num_sequences=num_sequences,
                generation_param_conf_path=generation_param_conf_path,
            )
        )
        for i, prediction in enumerate(predictions):
            print(f"\t# prediction-{i}: \t{prediction}")


def file_main(
    model_checkpoint_path: str,
    input_path: str,
    output_path: str,
    num_sequences: int,
    generation_param_conf_path: str,
    cuda_device: int,
    token_batch_size: int,
):

    model = load_pl_module_from_checkpoint(model_checkpoint_path)
    model.to(torch.device(cuda_device if cuda_device != -1 else "cpu"))
    model.eval()
    model.freeze()

    with open(input_path) as fi, open(output_path, "w") as fo:
        for source, sample_translations, _ in translate(
            model,
            map(lambda l: l.strip(), fi),
            num_sequences=num_sequences,
            generation_param_conf_path=generation_param_conf_path,
            token_batch_size=token_batch_size,
            progress_bar=True,
        ):
            for translation in sample_translations:
                fo.write(f"{source}\t{translation.strip()}\n")


def main():
    args = parse_args()
    if args.t:
        interactive_main(
            args.model_checkpoint,
            num_sequences=args.n,
            generation_param_conf_path=args.g,
            cuda_device=args.cuda_device,
        )
    else:
        file_main(
            args.model_checkpoint,
            args.f,
            args.o,
            num_sequences=args.n,
            generation_param_conf_path=args.g,
            cuda_device=args.cuda_device,
            token_batch_size=args.token_batch_size,
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_checkpoint", type=str, help="Path to pl_modules checkpoint")
    parser.add_argument("-n", type=int, default=1, help="Num sequences")
    parser.add_argument("-g", type=str, help="Path to generation conf")
    parser.add_argument("--cuda-device", type=int, default=-1, help="Cuda device")
    # interactive params
    parser.add_argument("-t", action="store_true", help="Interactive mode")
    # generation params
    parser.add_argument("-f", type=str, default=None, help="Input file")
    parser.add_argument("-o", type=str, default=None, help="Output file")
    parser.add_argument("--token-batch-size", type=int, default=128, help="Token batch size")
    # return
    return parser.parse_args()


if __name__ == "__main__":
    main()
