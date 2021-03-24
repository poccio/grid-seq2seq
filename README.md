<h1 align="center">
  Grid - Seq2Seq
</h1>

<p align="center">
  <a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-orange?style=for-the-badge&logo=pytorch"></a>
  <a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-blueviolet?style=for-the-badge"></a>
  <a href="https://hydra.cc/"><img alt="Config: hydra" src="https://img.shields.io/badge/config-hydra-blue?style=for-the-badge"></a>
  <a href="https://black.readthedocs.io/en/stable/"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-black.svg?style=for-the-badge"></a>
</p>

<h3 align="center">
  A simple template to bootstrap your Seq2Seq project
</h3>

Grid-Seq2Seq (Get RID of boilerplate code for Seq2Seq) is a simple and generic template to bootstrap your next [PyTorch](https://pytorch.org) project on generative 
Natural Language Processing models.
Quickly kickstart your work and get the power of:
* PyTorch and PyTorch Lightning (no comments needed here)
* Hydra, allowing to configure everything of your training in readable .yaml files
* A modular skeleton that supports several frequent use cases when dealing with generative NLP models:
    * HuggingFace [Transformers](https://github.com/huggingface/transformers) integration
    * Easily swap between generative models
    * Monitor autoregressive metrics such as bleu/rouge during training
    * Log generations inside [WandB](https://wandb.ai) tables during training
    * Interactive and file-based translation
  
If you are in a rush and have **2 minutes only**, be sure to check out the [quick tour](#quick-tour). 
When all else fails, read the - rest of the - instructions (a.k.a. the [FAQs](#faq-and-use-cases)).

If you use this template, please add the following shield to your README:
<br>
[![](https://img.shields.io/badge/-Grid--Seq2Seq-blueviolet?style=for-the-badge&logo=github)](https://github.com/poccio/nlp-gen)

## Quick Tour

A quite frequent scenario in the NLP community these days: you have a parallel dataset (usually in a TSV format),
representing some revolutionary way to use the latest advances in generative models, and you want to fine-tune a pretrained
encoder-decoder such as Bart on it. You are in the right place!

If you simply run the following:
```bash
PYTHONPATH=$(pwd) python src/scripts/model/train.py \
  exp_name=<exp-name> \
  project_name=<project-name> \
  data=simple \
  data.datamodule.train_dataset.path=<train-path> \
  data.datamodule.validation_dataset.path=<validation-path> \
  device=cuda
```
You get your Bart model, magically fine-tuned on the provided parallel dataset and everything logged on WandB, 
under <project-name>/<exp-name>. Once the training finishes (and eons have passed), 
you can use the interactive translate script to debug/present simple demos:
```bash
PYTHONPATH=$(pwd) python src/scripts/model/translate.py \
  <path-to-ckpt> -t -n 1 -g configurations/generation/beam.yaml
```
Note the script also support the common file-based translation.

However, **the main purpose of Grid-Seq2Seq is providing a template** to bootstrap your own project. 
As such, it is quite easy to change the various components that make up the experiment to match your requirements:
* Generative models are wrapped behind the [GenerativeModel] interface. As far as your models implement this interface, 
  you can easily swap among different models
* You can implement your Dataset/Datamodules and, as far as you are compliant with your generative model, everything will
  work transparently
  
Additionally, we are quite proud of the callbacks system. It allows you to log, inside nice WandB tables, the generations of your model as the training
progresses and, furthermore, you can use such generations to compute both referenced and unreferenced metrics upon them.

## Grid-Seq2Seq is your library for ...

* :rocket: Quick prototyping of generative models
* :skull: Modular skeleton to bootstrap your own project
* :telephone: Callback system! 
  * Use referenced metrics such as BLEU/Rouge as validation metric
  * Check the generations of your model as training goes on

## Template Structure

```bash
.
├── configurations
│   ├── generation                # hydra generation config files
│   └── hydra-train               # hydra config files for training
│       ├── root.yaml
│       ├── callbacks
│       ├── data
│       ├── device
│       ├── model
│       └── training
├── data                          # data folder
├── experiments                   # experiments folder
├── src       
│   ├── callbacks                 # callbacks code
│   ├── data                      # datasets and lightning datamodules
│   ├── generative_models         # supported generative models wrapped behind an interface
│   ├── optim                     # optimizers' instantiation code and custom optimizers
│   │   ├── factories.py
│   │   ├── optimizers
│   ├── pl_modules                # lightning modules
│   ├── scripts                   
│   │   ├── model                 
│   │   │   ├── train.py          # training script
│   │   │   └── translate.py      # translation script (both interactive and file mode supported)
│   └── utils
├── README.md
├── requirements.txt
└── setup.sh                      # bash script to auto-setup the env
```

## Setup Env

To neatly setup the whole environment needed for the project (interpreter, requirements, runtime dependencies, ...), 
we made a bash script that automatically configures everything. The only actual requirement is that you have [conda](https://docs.conda.io/projects/conda/en/latest/index.html)
installed. Once you have it installed (or if you already have it), just run the script and follow the prompts (desired python version, desired cuda version, ...) to quickly setup everything:

```
bash setup.sh
```

## Usage Examples

### Train a Summarization Model in 10 seconds

We chose [Summarization with Bart](https://www.aclweb.org/anthology/2020.acl-main.703.pdf) on the CNN/DailyMail dataset as our implemented working example.

As we mentioned earlier, everything is quite modular and, in order to carry out your given experiment, you just need to:
* Implement the various building blocks (mainly model and datamodule to use)
* Write the hydra configuration files
* Tell the training script to use them

In this case, we already took care of the first two steps; thus, we can directly jump to the actual training
(if you are not familiar with Hydra, we recommend reading Hydra [intro tutorials](https://hydra.cc/docs/tutorials/intro) to quickly get acquaninted with it),
launching the training script with arguments that instruct Hydra what components to use:

```bash
PYTHONPATH=$(pwd) python src/scripts/model/train.py \
  exp_name=bart-sum \
  project_name=<project-name> \
  data=cnn_dm \
  model=bart \
  callbacks=summarization \
  device=cuda_amp
```

Once the training finally finishes (and eons have likely passed), check out the translation script. In particular, besides
the common file-based mode, it features an interactive mode that can be useful for debugging/demos with your colleagues:

```bash
PYTHONPATH=$(pwd) python src/scripts/model/translate.py \
  <path-to-ckpt> -t -n 1 -g configurations/generation/beam.yaml
```

## FAQ and Use Cases

**Q**: I want to use another Generative Model. How?

**A**: It depends. If your model is part of *HuggingFace Transformers*, then you're golden. You just need to wrap it
behind the GenerativeModel interface and, if needed, write a suitable matching Dataset (or override some parts such as the
encoding). Consider the case of adding GPT2:
* You need to write a Dataset tailored for causal language modelling
* You need to write your GenerativeModel

Once you do, everything (callbacks, training, translations scripts) will work seamlessly.

Conversely, if your model is not part of *HuggingFace Transformers*, you may need to refactor part of the code: for example,
we currently have an explicit coupling in the training script toward *HuggingFace Transformers* Tokenizer object.
We welcome contributions in this direction.

**Q**: I want to monitor BLEU during training. How?

**A**: Check how we log Rouge (src.callbacks.generation.RougeGenerationCallback); it's essentially identical to that.
Once you have implemented your GenerationCallback, you just need to add it to your Hydra callback configuration file.

## Contributors

* [Edoardo Barba](https://github.com/edobobo)
* [Luigi Procopio](https://github.com/poccio)
