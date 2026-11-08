# Fine-tuning Qwen to read credit card transactions

The goal is to answer **“How much did I spend last month, by category, and which charges need my review?”** on my own computer, with real statements staying local. This course experiment tests one part of that workflow: teaching a small model to read a transaction description and return a merchant, amount, and category. It uses only synthetic data. Importing real statements and producing monthly reports are planned next steps.

[Open in Colab](https://colab.research.google.com/github/Shrezz/credit-card-llm-finetuning/blob/main/notebooks/Finetuning_Practice.ipynb) · [View the executed notebook](notebooks/Finetuning_Practice.ipynb) · [Sample: one transaction per category](data/sample.jsonl) · [Frozen synthetic dataset](data/transactions_v1.zip)

## Why fine-tune for this task?

The output has three fixed fields and eleven allowed categories. What the model needs to learn is how to apply a labeling convention to different descriptions. For example, `TARGET #1284` gets `Other / Mixed`, while `TARGET OPTICAL #1284` gets `Healthcare`. [LoRA](https://arxiv.org/abs/2106.09685) lets us teach that behavior from labeled examples while keeping the original model weights frozen. We tested whether that helped by giving both models the same prompt and the same unseen transactions.

**When would retrieval help?** [Retrieval-augmented generation (RAG)](https://arxiv.org/abs/2005.11401) gives the model reference material at prediction time. That would be useful if a category depended on a changing expense policy or a merchant directory. This experiment has no such reference source. A directory might identify a Target store, but it still couldn't tell us what was in the basket. A user correction for a known merchant is simpler to keep as a local lookup rule.

**What the evidence supports.** With the prompt alone, 0/66 raw replies parsed as JSON; with the adapter, 66/66 parsed and 38/66 had every field correct. That establishes an improvement over the tested prompt. We did not benchmark RAG, a conventional classifier, or a decoder that enforces JSON syntax. The [results](#results) also show where merchant extraction regressed.

**How this fits expense analysis.** In the planned local tool, Python will read dates and amounts from the statement, apply saved merchant rules, and calculate totals. The model will suggest categories for unfamiliar descriptions, and uncertain charges can be reviewed. A $50 purchase, a $10 refund, and a $500 card payment should yield **$40 in net spending**: the payment settles the balance and must not be counted as another purchase. Getting that arithmetic right is a separate test from getting a category right.

## A prediction from the test set

Input: `BHARAT BAZAAR / PURCHASE 10/14 / $47.60`

Actual fine-tuned reply:

```json
{"merchant": "Bharat Bazaar", "amount": 47.60, "category": "Groceries"}
```

The task is to identify the merchant, copy the amount as a number, and assign a spending category. A grocery label says what kind of merchant the statement identifies; it does not tell us which items were bought. For a general retailer with no narrower description, the labeling rule is `Other / Mixed`.

## Results

Both runs used the same 66 test rows, prompt, tokenizer, and greedy decoding settings. Each score below uses **all 66 rows** as its denominator.

| Metric | Original Qwen | Qwen + LoRA |
|---|---:|---:|
| Category accuracy | 0/66 (0.0%) | 50/66 (75.8%) |
| Merchant accuracy | 0/66 (0.0%) | 42/66 (63.6%) |
| Amount accuracy | 0/66 (0.0%) | 66/66 (100.0%) |
| JSON validity | 0/66 (0.0%) | 66/66 (100.0%) |
| All three fields correct | 0/66 (0.0%) | 38/66 (57.6%) |

**What the baseline zero means.** Every original reply wrapped its JSON in Markdown fences. The strict scorer tried to parse the entire reply, so none earned field credit. To distinguish formatting from content, I also checked what happened when removing *only* the opening and closing fences from those saved replies. All 66 then parsed: 47 merchants, 32 amounts, 32 categories, and 10 complete transactions were correct. This diagnostic is separate from the official scores. It shows a real tradeoff: the adapter improved formatting, amounts, and categorization, but matched **five fewer merchant names** once formatting was accounted for.

Two errors show what the aggregate scores miss:

| Statement excerpt | Expected | Fine-tuned prediction | What happened |
|---|---|---|---|
| `LAKEVIEW MARKET #12` | Lakeview Market · Groceries | Lakeview Market · Shopping | It removed the store number correctly, but chose a different category from the catalog. |
| `WILLOW BISTRO` | Willow Bistro · Dining | Willy Bistro · Groceries | It changed a merchant name that was already present and missed the category clue in “Bistro.” |

## Data and method

All 396 transactions were generated from a [synthetic merchant catalog](data/merchant_catalog_v1.json), statement templates, seeded dates, and amounts drawn in whole cents. No real financial records were used. Each merchant had three statement aliases. Merchant families, their aliases, and the statement templates were assigned to splits before generating rows.

| Split | Merchants | Formats | Rows | Purpose |
|---|---:|---:|---:|---|
| Training | 44 | 2 | 264 | Update the adapter |
| Validation | 22 | 1 | 66 | Measure loss before and after training |
| Test | 22 | 1 | 66 | Compare generated replies on the same frozen examples |

The splits are balanced across 11 categories: Groceries, Dining, Shopping, Transportation, Gas, Entertainment, Healthcare, Travel, Utilities, Subscriptions, and Other / Mixed. The initial Target and Vasona Park examples were development exercises and are excluded from the reported scores.

**Architecture.** The base model was [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B), loaded in FP16. LoRA kept the base weights fixed and trained small updates in the attention layers (`q_proj` and `v_proj`). Of 597,196,800 parameters in the model with its adapter, 1,146,880 were trainable (0.192%). This experiment used LoRA; the base was not quantized for QLoRA.

**Training.** TRL’s `SFTTrainer` scored the expected completion tokens, leaving the prompt out of the loss. Settings: LoRA rank 8, alpha 16, dropout 0.05; initial learning rate `1e-4`; batch size 2 with four gradient accumulation steps; maximum sequence length 256; seed 42. One epoch was 33 optimizer updates. Mean training loss was **0.2094**; validation loss fell from **0.8406 to 0.1912**. Loss measures how well the model predicts the next answer token given the preceding correct tokens. The test table measures what it actually generated on its own.

**Scoring.** The scorer parses the raw reply without removing fences or extra prose. An unparseable reply earns no field credit. For parsed objects, merchant names ignore case and surrounding spaces, category labels must match exactly, and amounts must be JSON numbers equal in decimal value (`8` equals `8.00`; `650` does not equal `6.50`).

## Reproduce the experiment in Colab

1. Open the notebook linked above and connect a GPU runtime. The experiment pins `transformers==4.56.2`, `peft==0.17.1`, and `trl==0.23.1`. Run the dependency setup before loading the model.
2. Upload `transactions_v1.zip` using Colab’s Files panel. Use this ZIP directly to reproduce the scored run; skip cells that generate or download a new ZIP. To recreate the data instead, use the notebook’s generator with `merchant_catalog_v1.json`, the original formats, and seeds 42/43/44 for train/validation/test.
3. Run the prompt, model-loading, inference, and scoring definitions. Evaluate the **original model before attaching an adapter**. Read the frozen test rows, send only each row’s `input` to the model, and save the raw replies. The inference settings are thinking disabled, `do_sample=False`, and `max_new_tokens=100`.
4. Prepare prompt/completion pairs from `train.jsonl` and `validation.jsonl`. Attach a fresh LoRA adapter, evaluate starting validation loss, train for one epoch, and save the adapter and training log.
5. Put the trained model in evaluation mode. Run the same prediction and scoring functions on the same test rows. Keep the original and fine-tuned JSONL files separate. The SHA-256 of the scored `test.jsonl` is `4a4b94dba23f908ddb5108f9806b3ec58302144406ab538f72aceb9ff97963ae`.

To inspect the recorded results without a GPU, open the executed notebook on GitHub. Its final cells show the measured scores and individual errors. Rerunning the notebook in Colab creates `evaluation_results.zip` (both prediction files, metrics, and comparison) and `transaction_lora_adapter.zip` (adapter weights, configuration, and training log).

## Limitations

The test set has six transactions per category from two merchants per category; all test rows use the same held-out statement format. Several rows are aliases of the same merchant, so they are not 66 independent merchant tests. A name such as `LAKEVIEW MKT` may not tell a reader whether the catalog treats it as a grocery store or a mixed retailer. These results describe one training run on synthetic purchases with positive amounts. They do not establish accuracy on real statements, refunds, or other currencies.
