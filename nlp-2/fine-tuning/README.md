# Fine-tuning: LoRA and prompt tuning

Lab 4 of NLP II. Parameter-efficient fine-tuning modules for `google/flan-t5-small`, used in the course for dialogue summarization.

## Implemented

`src/finetuning.py`
- `LoRA`: wraps a linear layer with a trainable low-rank update (matrices A and B, rank `r`, scaling `alpha / r`) and freezes the original layer.
- `inject_lora_into_model`: replaces the attention projection layers of a model with `LoRA` layers.
- `SoftPromptEmbedding`: trainable virtual tokens prepended to the input embeddings.

The notebook that ran full fine-tuning, LoRA and prompt tuning on DialogSum belongs to the course material and is not included, so no evaluation results are reported here.

## Run

```
pip install -r requirements.txt
cd src
python -c "import finetuning"
```
