This is a simple finetuning script for the omniASR LLM-ASR model. It uses the LoRA technique to finetune the model on a specific dataset. The dataset I chose is the Casablanca dataset from the UBC-NLP dataset. If you wanted to use your own dataset, you would have to change TrainingConfig, the dataset class (currently CasablancaOmniASRDataset) and the collator (currently OmniASRCollator).

The load_llm_asr function currently uses the 300M LLM-ASR variant by default. You can choose another size by specifying the model_size parameter (one of "300m", "1b", "3b", "7b").

You can also change the model to one of the other variants by changing the load_llm_asr function and also changing the code for printing some examples during the training run.

I had to create my own implementation of LoRa because the peft library is not compatible with the omniASR model as it uses fairseq2 for the linear layers instead of torch.nn.Linear.

You can modify which parts of the model to finetune by changing the target_keywords in the LoraConfig class.

For inference, you can use the load_llm_asr_300m_with_lora function to load the model with the LoRA weights.

Note, that the saved checkpoint will only contain the LoRA weights. You'll have to load the base model weights separately.

The current implementation does an evaluation step every 35 steps. You can change this by changing the eval_every_steps parameter in the TrainingConfig class.

Note, that I have added some packages that I use during training that you'll have to install.

- wandb
- datasets
- tqdm

running the script from the root of the repository:

```bash
python workflows/recipes/finetune_lora/finetune.py
```
