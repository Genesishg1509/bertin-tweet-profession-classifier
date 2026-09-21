# ProfNER Profession-Mention Classifier — BERTIN vs. a TF-IDF Baseline

Does a tweet mention a healthcare profession? This is a binary text classification task from [ProfNER](https://temu.bsc.es/smm4h-spanish), a shared task built around tweets from the COVID-19 pandemic. The dataset ([`luisgasco/profner_classification_master`](https://huggingface.co/datasets/luisgasco/profner_classification_master) on Hugging Face) has 2,786 training tweets, 999 for validation, and 1,001 unlabeled test tweets to predict.

This started as a university NLP exercise. Fine-tuning **BERTIN**, a RoBERTa model pretrained specifically on Spanish, got F1 ≈ 0.88. This repo restructures that work into something reproducible, adds a baseline to check whether the transformer's cost is justified, and fixes a bug in how the decision threshold was applied.

## The problem, the technique, and why

**Problem:** there isn't a real-world problem to solve here — this is a university project, built on a fixed academic benchmark (a shared task with a public leaderboard). It's closer to research than to product work: an exercise to practice and demonstrate NLP skills, not to answer a business question.

**Technique:** fine-tuning BERTIN (a RoBERTa model pretrained on Spanish) for binary text classification, benchmarked against a TF-IDF + Logistic Regression baseline, with a threshold bug from the original coursework found and fixed.

**Why:** the baseline comparison, the bug fix, and the honesty about the threshold caveat (below) are the parts that turn a one-off coursework notebook into something closer to how this kind of model would actually be validated before anyone trusted it. (Why BERTIN specifically, as opposed to another transformer, is covered further down.)

## Is the transformer worth it?

BERTIN is a 110M-parameter model that needs a GPU to fine-tune in reasonable time. Before paying that cost, it's worth checking what a much cheaper model can do: TF-IDF features into logistic regression, which trains in seconds on a laptop with no GPU.

| Model | F1 | Accuracy | Training cost |
|---|---|---|---|
| TF-IDF + Logistic Regression | 0.547 | 0.702 | Seconds, CPU |
| **BERTIN RoBERTa (fine-tuned)** | **0.878** | **0.942** | ~15 min, GPU |

The gap is large — 33 points of F1. Unlike cases where a simple baseline nearly matches a complex model, here it clearly doesn't. That's a real answer to "is this complexity worth it," and it's yes: the transformer picks up on patterns in how professions are mentioned that a bag-of-words model can't capture.

*(TF-IDF result reproduced in this environment on 2026-09-14. BERTIN's result is from the original training run — reproducing it requires a GPU, which wasn't available while building this repo. See "Reproducing this" below.)*

## A bug in the original notebook

The original analysis searched for the F1-optimal decision threshold on the validation set (0.42 instead of the default 0.5) — but then generated the final test predictions using the model's raw argmax, which is the same as always using threshold 0.5. The tuned threshold was computed and then never used.

`src/evaluate.py` fixes this: `find_best_threshold` tunes it, and `src/predict.py` applies that same threshold when generating the actual submission file.

**An honest caveat about this threshold, stated plainly:** it's tuned on the same ~1,000-example validation set used to report F1. That's optimistic — it's the best threshold *for this validation set*, not a guarantee on new data. A more rigorous setup would tune it on a separate held-out split. This repo doesn't have enough data to afford that split and still leaves the caveat visible rather than hiding it.

## Project structure

```
src/
  preprocess.py   Text cleaning + dataset loading
  baseline.py     TF-IDF + Logistic Regression
  train.py        Fine-tune BERTIN, tune the threshold on validation
  evaluate.py     Shared metrics + threshold search (with the honesty caveat above)
  predict.py      Apply the tuned threshold to generate final test predictions

app.py            Gradio demo -- paste a tweet, get a prediction
notebooks/        Cleaned analysis notebook (EDA, no course-specific scaffolding)
reports/          Metrics as JSON, for anything that reads them programmatically
```

## Reproducing this

```bash
pip install -r requirements.txt

python -m src.baseline              # TF-IDF baseline, runs on CPU in seconds
python -m src.train                 # fine-tunes BERTIN -- needs a GPU (Colab's free tier works)
python -m src.predict --threshold 0.42   # generates reports/predicciones_test.tsv
python app.py                       # local Gradio demo (needs a trained model in models/bertin)
```

## Why BERTIN specifically

BERTIN RoBERTa is pretrained on Spanish text drawn from mC4, covering both Latin American and Iberian Spanish, rather than being a multilingual model fine-tuned as an afterthought. RoBERTa's architecture tends to do well on classification tasks, and BERTIN integrates directly with the Hugging Face ecosystem used throughout this project.

## License

MIT.
