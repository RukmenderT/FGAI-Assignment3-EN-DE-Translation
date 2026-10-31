"""Post-run output checks on the 1,000 test translations.

For every system this counts outputs that copy the English source (untranslated),
outputs that are far too short or too long, and BLEU with the copied outputs removed.
Run from the repository root:  python analysis/output_checks.py
Writes metrics/output_checks.csv
"""
import pandas as pd
import sacrebleu

SYSTEMS = {
    "transformer": "Basic Transformer",
    "mt5": "mT5-small",
    "llama_zeroshot": "LLaMA-3.2-1B-Instruct (zero-shot)",
    "llama_lora": "LLaMA-3.2-1B-Instruct + LoRA",
    "gemma_zeroshot": "Gemma-2-2B-it (zero-shot)",
    "gemma_lora": "Gemma-2-2B-it + LoRA",
}

test = pd.read_csv("data/test_newstest2016_sample.tsv", sep="\t", keep_default_na=False)
src, ref = test["en"].tolist(), test["de"].tolist()
rows = []
for key, name in SYSTEMS.items():
    hyp = open(f"predictions/{key}.txt", encoding="utf-8").read()[:-1].split("\n")
    assert len(hyp) == len(src), f"{key}: {len(hyp)} lines"
    ratio = [len(h.split()) / max(1, len(s.split())) for h, s in zip(hyp, src)]
    copied = {i for i, (h, s) in enumerate(zip(hyp, src)) if sacrebleu.sentence_chrf(h, [s]).score > 70}
    keep = [i for i in range(len(src)) if i not in copied]
    rows.append({
        "System": name,
        "Untranslated (copied English)": len(copied),
        "Too short (<50% of source length)": sum(r < 0.5 for r in ratio),
        "Too long (>200% of source length)": sum(r > 2.0 for r in ratio),
        "BLEU (all 1,000)": round(sacrebleu.corpus_bleu(hyp, [ref]).score, 2),
        "BLEU (copies removed)": round(sacrebleu.corpus_bleu([hyp[i] for i in keep], [[ref[i] for i in keep]]).score, 2),
    })

out = pd.DataFrame(rows)
out.to_csv("metrics/output_checks.csv", index=False)
print(out.to_string(index=False))
