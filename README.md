# CS781 Capstone: In-House Legal Intelligence System (Legal Language Model from Scratch)

**Author:** Akshat Kasarwal | **Roll No:** 240085 | **Codabench user:** `a_240085`
**Course:** CS781, IIT Kanpur | **Instructor:** Prof. Ashutosh Modi

This repository holds the source code, notebooks, and submissions for my individual CS781 capstone. The scenario: a law firm bound by confidentiality cannot send client documents to external LLM APIs, so we build a **self-hosted legal NLP stack** from compact models **trained from scratch** on the official Indian legal corpus ([`Exploration-Lab/CS-781-Capstone`](https://huggingface.co/datasets/Exploration-Lab/CS-781-Capstone)), then adapt them to downstream legal tasks. The benchmark follows [IL-TUR](https://arxiv.org/abs/2407.05399).

**Constraints:** 50M-300M parameters, no pretrained weights, no external legal corpora, no external LLM APIs, Kaggle T4x2 sessions.


---

## 1. Results

Ranks recorded on 9 November 2026. 110 valid Codabench submissions in total (excluding failed and starter runs).

| Phase | Task | Primary metric | Best score | Rank | Finished subs. |
|---|---|---|---|---|---|
| 1 | Legal LM pretraining | BERTScore F1 | **0.8279** | 103 | 8 |
| 2 | Legal NER | Strict Macro-F1 | **0.5547** | 43 | 25 |
| 3 | Rhetorical role prediction | Macro-F1 | **0.4742** | 84 | 40 |
| 4 | Court judgment prediction | Macro-F1 | **0.6915** | 83 | 25 |
| 5 | Legal statute identification | Macro-F1 | not attempted | n/a | 0 |
| 6 | Abstractive summarization | BERTScore F1 | **0.8186** | 117 | 6 |
| 7 | Prior case retrieval | Micro-F1@7 | **0.2416** | 120 | 6 |


---

## 2. Repository structure

```
CS781-Legal-Language-Model-from-Scratch/
├── Phase 1/   # Tokenizer + from-scratch pretraining of 4 architectures, decoding, submissions
├── Phase 2/   # Legal NER (BIO tagging, Viterbi decoding, 5-fold CV, ensembles)
├── Phase 3/   # Rhetorical role prediction (sentence-window classifier, 4 backbones)
├── Phase 4/   # Court judgment prediction (binary, long-document views)
├── Phase 6/   # Abstractive summarization (prompted decoder LM)
├── Phase 7/   # Prior case retrieval (contrastive dual-encoder style)
└── README.md
```

---

## 3. Phase 1: Legal language model pretraining

### 3.1 Tokenizer (shared by all four models)
- Byte-level BPE, **12,000 tokens**
- Learned on the first 8M characters from the 100,000 most frequent word types
- Whitespace pre-tokenization, `</s>` emitted after every word (about 2.7 tokens per word, so 512 tokens cover only about 150-250 words)

### 3.2 Architectures
| Model | Description |
|---|---|
| **Decoder-only Transformer** | Rotary embeddings (RoPE), multi-query attention (12 query heads, 1 KV head), RMSNorm, SwiGLU |
| **Conv-Attention Hybrid** | Gated causal depthwise-convolution blocks plus global rotary-attention blocks (3:1 ratio, kernel 15) |
| **Retention (RetNet-style)** | Retention mechanism in place of softmax attention |
| **Encoder-Decoder** | Prefix-LM style, 6+6 layers, cross-attention |

### 3.3 Training configuration
Shared: AdamW (beta = (0.9, 0.95), weight decay 0.1), grad-clip 1.0, fp16, dropout 0.1, warmup then cosine decay.

| Model | Params | Context | Peak LR | Tokens/step | Steps | Tokens seen | Val. ppl |
|---|---|---|---|---|---|---|---|
| Decoder | 109.5M | 512 | 3e-4 | 4.1k | 45k | 184M | 4.17 |
| Hybrid | 103.1M | 1024 | 3e-4 | 8.2k | 7k | 57M | 4.82 |
| Retention | 73.6M | 1024 | 5e-4 | 16.4k | 12k to 20k | 196M (at 12k) | 5.07 |
| Enc.-Dec. | 96.2M | 512+400 | 3e-4 | 7.3k | 7.5k | 55M | 7.68 |

> Perplexities are **not comparable across rows** (different contexts and validation sets; the encoder-decoder is scored on target tokens only).

- Data: 300M-token training cache (about 6.7% of the corpus); 10M-token validation cache; 200 documents held out, 74 of which gave a full 512-word prompt and were used for a decoding grid.
- The decoder was trained 7.5k, resumed to 15k, then 45k steps (val loss 1.7 to 1.43; about 5.2 h on T4x2). The retention model was resumed from 12k to 20k with a warm-restart LR and no dropout.
- **Decoding (submitted):** temperature 0.5, top-p 0.9, top-k 50, repetition penalty 1.2. The other three models used default sampling (temperature 0.8, top-p 0.95).
- An early generation routine truncated prompts to about 112 tokens; this bug was found and fixed.
- Local BERTScore (RoBERTa-large, 0.857-0.861) is **not comparable** with Codabench; its 0.004 spread is too small to rank decoding settings.

### 3.4 Codabench results
| Model | Steps | BERTScore F1 |
|---|---|---|
| Decoder | 7.5k | 0.8188 |
| Decoder | 15k | 0.8245 |
| **Decoder** | **45k** | **0.8279** |
| Hybrid | 7k | 0.8243 |
| Retention | 12k | 0.8261 |
| Retention | 20k | 0.8278 |
| Enc.-Dec. | 7.5k | 0.8142 |

---

## 4. Transfer protocol (Phases 2-7)
- All downstream models are initialised from Phase 1 checkpoints with a task head; no other pretrained model is used.
- Phase 1 weights, embeddings and tokenizer are reused unchanged; the LM head is replaced. For classification/tagging the causal mask is switched off unless stated.
- **Backbones per phase:** Phases 2 and 3 on all four models; Phase 4 on the decoder, hybrid and encoder-decoder; Phases 6 and 7 on the decoder only.
- **Techniques compared:** full and partial fine-tuning, BitFit, LoRA, LoRMA+, AdaLoRA, Houlsby/Pfeiffer adapters, IA3, prompt tuning, prefix tuning, knowledge distillation (subset depends on phase and architecture; prefix tuning was not applied to retention/hybrid, which have no KV cache).
- **KD:** 6-layer student (59.4M params in Phases 2-4 and 6); teacher is the fully fine-tuned model (Phases 3 and 7) or the LoRMA run (Phase 6).
- **Compute:** free Kaggle T4x2, wall-clock-capped runs (Phase 2 folds about 17-20 min, Phase 3 8-24 min, Phase 4 75 min, Phase 7 15 min).

---

## 5. Phase 2: Legal NER

**Model:** decoder trunk (causal mask off) with a linear BIO token-classification head over the 12 labels (APP, RESP, A.COUNSEL, R.COUNSEL, JUDGE, WIT, AUTH, COURT, STAT, PREC, DATE, CASENO).

**Pipeline**
- Words split into word and punctuation pieces; every BPE token keeps its character offsets.
- 512-token sliding windows with stride 384; BIO-constrained **Viterbi** decode over window-averaged log-probabilities, mapped back to exact character offsets. Oracle check: strict Macro-F1 0.995 (train), 0.999 (dev).
- Loss down-weights `O` (0.5) and keeps half the entity-free windows per epoch.
- **Validation:** the public dev split has only 4 documents, so every method was trained as **5-fold CV over all 70 labelled documents**, scored out-of-fold; the 5 fold models are averaged on the 35 test documents.
- **v3 additions:** causal-attention LoRA/LoRMA+ counterparts (rank 16 on all attention and FFN projections, about 3.1M / 3.3M trainable, LR 1e-3), a second CV round (rank 32, stride 256, O-weight 0.7, LR 5e-4, 10 epochs), boundary trimming of edge punctuation, and 3 seeds of final models trained on all 70 documents for 8 epochs.
- **LoRMA+ form used:** multiplicative variant `y = h + (alpha/r) B(Ah)` with `h = W0 x`, initialised to identity.
- **Final submission:** `v3 cv + fin` ensemble = 30 fold models + 6 final models (final models weighted 2).
- **Rejected:** per-label confidence thresholds (0.5396 on all out-of-fold docs, but a split-half check lost 0.006 on average).

**Codabench (Strict Macro-F1)**

| Single run | Score | Ensemble | Score |
|---|---|---|---|
| LoRA | 0.5079 | Decoder LoRA | 0.5466 |
| LoRMA | 0.5003 | Decoder LoRMA | 0.5386 |
| Full (causal) | 0.4848 | Decoder all methods | 0.5413 |
| Full (bidirectional) | 0.4705 | Decoder KD | 0.4885 |
| KD | 0.4469 | Retention (LoRA/LoRMA) | 0.4862 / 0.4869 |
| Adapter | 0.3812 | Hybrid (LoRA/LoRMA) | 0.4856 / 0.4867 |
| Partial | 0.3722 | Enc.-Dec. (LoRA/LoRMA) | 0.4654 / 0.4703 |
| IA3 | 0.3042 | **v3 CV all / CV+final** | 0.5524 / **0.5547** |

**Findings:** bidirectional beats causal on out-of-fold Macro-F1 for LoRA (0.4947 vs 0.4480) and LoRMA+ (0.4980 vs 0.4544), but single-split full fine-tuning on 35 test docs pointed the other way (0.4848 causal vs 0.4705 bidirectional); the 70-document comparison is more trustworthy. Hardest labels: APP (OOF F1 0.17), RESP (0.21), WIT (0.22), because names rarely repeat (142, 110, 58 distinct strings). AUTH has precision 0.47 but recall 0.26. Easiest: DATE (0.955), COURT (0.794).

---

## 6. Phase 3: Rhetorical role prediction

**Model:** documents are cut into windows of whole consecutive sentences (at most 512 tokens, each sentence prefixed by `<s>`, truncated to 128 tokens). The decoder trunk runs bidirectionally; token states are mean-pooled per sentence; a linear 13-way head classifies `[pooled vector ; relative-position embedding ; CL/IT domain embedding]`. At inference each sentence is predicted once, inside a core of at most 256 tokens padded with neighbouring sentences up to 512.

**Training:** CL and IT trained jointly; class-weighted cross-entropy (weight proportional to freq^-0.5, capped at 10); checkpoint chosen on Macro-F1 over the 10 dev documents; LR 1e-3 (head/adapter), 8 epochs, 100 warmup steps.

**Methods (decoder):** full FT, LoRA, LoRMA+ (rank 16, alpha 32, about 3.3M trainable), AdaLoRA (init rank 24, target 16), IA3, prompt tuning (32 tokens), prefix tuning (32 positions, MLP re-param), KD into a 6-layer student (59.4M, T=2, alpha=0.5). Same recipe repeated on retention, hybrid and encoder-decoder.

**Codabench (Macro-F1), best run per method and backbone**

| Method | Decoder | Retention | Hybrid | Enc.-Dec. |
|---|---|---|---|---|
| Full | 0.4461 | 0.4502 | n/a | n/a |
| LoRA | 0.4476 | 0.4520 | 0.4143 | 0.3316 |
| LoRMA | 0.4371 | 0.4265 | 0.4205 | n/a |
| AdaLoRA | 0.4288 | 0.4221 | 0.4189 | 0.2976 |
| IA3 | 0.4266 | 0.4105 | n/a | n/a |
| KD | 0.4363 | 0.4464 | 0.4220 | n/a |
| Prompt | 0.3354 | 0.4104 | n/a | 0.2544 |
| **Ensemble** | 0.4665 | **0.4742** | 0.4289 | 0.3910 |

**Caveats:** the dev set has 10 documents and contains **no Dissent or PrecedentOverruled** sentences, so dev Macro-F1 is capped at 11 of 13 classes and 0.01-0.02 differences are noise. One greedy ensemble and a per-class bias were fitted on those 10 dev documents and are therefore optimistic. No per-class breakdown was computed.

---

## 7. Phase 4: Court judgment prediction

**Model:** decoder (bidirectional) encodes a 512-token chunk, mean-pool, dropout, linear 2-way head. Text is **lower-cased** (train is cased, dev/test are lower-cased). Only the first 150 and last 1,000 words are tokenized; the base setting uses the **last 512 tokens** (judgments end with the operative order). Four-chunk (attention pooling), head, and head+tail views were tried with LoRMA only.

**Training:** class-weighted CE (freq^-0.5), batch 4 x 2 accumulation, head LR 1e-3, decision threshold on the logit margin **calibrated on the balanced dev set** (so dev scores are optimistic). Train set is 41.0% positive; dev/test are about 50%.

**Techniques:** full FT (5e-5, 2 epochs), partial FT (top 4 blocks + final norm, 33.4M trainable, LR 1e-4), BitFit (34k params), Houlsby adapters (2.4M), LoRMA+ (3.3M), AdaLoRA (4.7M), KD into a 6-layer 59.4M student. Greedy ensemble averages full, AdaLoRA and KD. Hybrid reads last 1,024 tokens; encoder-decoder reads last 512.

**Codabench (Macro-F1), decoder, best per technique**

| Technique | Score |
|---|---|
| **Partial FT** | **0.6915** |
| Greedy ensemble | 0.6853 |
| KD | 0.6833 |
| Full FT | 0.6664 |
| AdaLoRA | 0.6321 |
| BitFit | 0.5855 |
| Adapter | 0.5849 |
| LoRMA+ | 0.5191 (failed, see below) |

**LoRMA+ failure:** in all four decoder runs (tail, head, head+tail, multi-chunk) training loss was `nan` throughout. The tail run predicts one class under argmax; the others predict class 0 for every case. The all-methods ensemble containing them is byte-identical to the all-zero output and scored 0.3323, so it says nothing about ensembling. The all-zero failure recurred on the encoder-decoder (3 of 5 LoRMA runs) and the hybrid (head+tail). Not diagnosed.

**Dev vs test gap:** partial FT scored 0.593 on dev but 0.6915 on test; AdaLoRA 0.654 on dev but about 0.59-0.63 on test. Dev (balanced, calibrated on) is not a reliable predictor of test here.

---

## 8. Phase 5: Legal statute identification

**Not attempted. No submission was made.**

---

## 9. Phase 6: Abstractive summarization

**Model:** the decoder is used as a prompted LM. Input = head and tail of the judgment (about 640 tokens, 70% from the start) + marker `SUMMARY:`; target = reference summary ending in `ENDSUM`; loss on summary tokens only (up to 384). To feed 1024-token sequences into a model pretrained at 512, **rotary positions are linearly interpolated by 0.5**. 150 training documents held out for validation.

**Setup:** 3 epochs per run (1 h cap), best checkpoint by lowest validation loss, batched KV-cache decoding. Three decoding settings (greedy; greedy + repetition penalty 1.15; sampling at T=0.5, top-k 50, top-p 0.9 + penalty) compared by BERTScore on 64 validation documents. LRs, ranks and epochs were notebook defaults and were **not searched**.

| Method | Config | Trainable (M) | % of backbone | BERTScore F1 |
|---|---|---|---|---|
| Houlsby adapters | bottleneck 64 after attn + FFN, + RMSNorm gains, LR 5e-4 | 2.40 | 2.19 | **0.8186** |
| LoRA | rank 16, alpha 32, all linear, LR 5e-4 | 3.12 | 2.85 | 0.8182 |
| KD (6-layer student) | layers 1,3,5,7,9,11 init; teacher = LoRMA; T=2, alpha=0.5 | 59.36 | 54.2 | 0.8172 |
| Prefix tuning | 48 positions via 512-unit MLP, LR 2e-3 | 1.08 | 0.98 | 0.8172 |
| LoRMA (permutation variant) | multiplicative low-rank + cyclic row rotation, rank 16, attention proj., LR 2e-4 | 0.64 | 0.58 | 0.8136 |
| Prompt tuning | 64 soft tokens, LR 5e-3 | 0.05 | 0.04 | 0.8128 |

Differences of about 0.001 between adjacent rows should not be ranked. The KD student trains all weights, which may explain why it beat its teacher.

---

## 10. Phase 7: Prior case retrieval

**Model (decoder only, from 45k-step weights):** each document is split into **480-token chunks (up to 8)**, encoded with causal attention, and mean-pooled into one vector. Training uses one random chunk per document; inference averages up to 8 chunks. Candidates ranked by **cosine similarity**, top 7 returned, TF-IDF score fusion **off**. Loss: **InfoNCE** (temperature 0.05) with in-batch negatives plus 2 TF-IDF-mined hard negatives per query (other gold documents masked). Each technique had a 15-minute wall-clock budget and its own LR (full FT 5e-5). Candidates capped at 3,840 tokens.

| Technique | Detail | Micro-F1@7 |
|---|---|---|
| **Full FT** | all 109.5M params | **0.2416** |
| LoRMA | multiplicative update (I + BA)W on Q, V, O; rank 16 | 0.2357 |
| Partial FT | last 3 blocks | 0.2262 |
| KD | 6-layer student (layers 0,2,4,6,8,11); contrastive + KL + cosine-embedding losses; teacher = full FT | 0.2191 |
| IA3 | n/a | 0.2085 |
| Prompt tuning | 32 learned tokens | 0.2079 |

No BM25 / lexical-only baseline was run, so it is unknown whether the neural model beats a trivial lexical retriever here.

---

## 11. Hyperparameters of the final model per phase

| Hyperparameter | P1 | P2 | P3 | P4 | P6 | P7 |
|---|---|---|---|---|---|---|
| Backbone params (M) | 109.5 | 109.5 | 109.5 | 109.5 | 109.5 | 109.5 |
| Trainable (M) | 109.5 | 3.1-3.3 | 3.3 | 33.4 | 2.4 | 109.5 (full) |
| Context (tokens) | 512 | 512 | 512 | 512 (tail) | 1024 | 480 x 8 |
| Peak LR | 3e-4 | 1e-3 | 1e-3 | 1e-4 | 5e-4 | 5e-5 (full) |
| Batch x accum. | 4 x 2 | 8 | 8 | 4 x 2 | 4 x 4 | 32 seq/step |
| Epochs / steps | 45k | 10 (CV), 8 (final) | 8 | 2 | 3 | 15 min |
| Dropout (adapter / head) | 0.1 | 0.05 / 0.1 | 0.05 / 0.1 | n/a / 0.1 | 0.1 | 0.1 |
| Decoding / threshold | T 0.5, p 0.9 | Viterbi | argmax | calibrated | grid of 3 | cosine, top 7 |

Phases 2-4 use 100 warmup steps, weight decay 0.01, head LR 1e-3; Phase 6 uses 5% warmup and no weight decay; Phase 1 uses weight decay 0.1.

---

## 12. Limitations and future work
- Pretraining used only about 4% of the available 4.5B tokens (184M for the decoder).
- Hyperparameters were not systematically tuned; each architecture got limited runs.
- Phase 5 missing; Phases 6 and 7 decoder-only.
- Tiny validation sets in Phases 2 and 3 limit the reliability of method comparisons.
- LoRMA+ numerical instability in Phase 4 undiagnosed.
- No human or legal-correctness evaluation; BERTScore/F1 do not establish that outputs are legally accurate. These models are experimental, not dependable legal assistants.
- **Next:** longer pretraining, systematic tuning, diagnose LoRMA+ (try fp32 / bf16-safe updates, smaller LR), add BM25 baseline for PCR, human evaluation.

---

## 13. Reproducibility notes
- Platform: Kaggle notebooks, 2x T4.
- Data: load `Exploration-Lab/CS-781-Capstone` from Hugging Face (phase configs `phase_1` ... `phase_7`). Only official data was used.
- Run order: Phase 1 notebooks first (tokenizer, then each architecture), then save the checkpoints; every later phase loads those Phase 1 weights.
- Submission format: `prediction.json` at the root of `submission.zip`.
- Seeds and exact notebook filenames are as stored in each phase folder.

---

## 14. LLM usage
LLMs used: Claude and ChatGPT. Used for explaining methods (e.g. LoRMA, retention), brainstorming ideas, debugging notebook/file-path errors, plotting code, drafting factual sections of the report, and **writing the core training/inference code for Phases 2, 3, 4, 6 and 7**.

---

## 15. References
1. Joshi et al., *IL-TUR: Benchmark for Indian Legal Text Understanding and Reasoning*, ACL 2024.
2. Zhang et al., *BERTScore*, ICLR 2020.
3. Bihany, Patel, Modi, *LoRMA: Low-Rank Multiplicative Adaptation for LLMs*, arXiv:2506.07621.
4. Hinton et al., *Distilling the Knowledge in a Neural Network*, arXiv:1503.02531.
5. Su et al., *RoFormer (RoPE)*, arXiv:2104.09864.
6. Shazeer, *Fast Transformer Decoding: One Write-Head Is All You Need*, arXiv:1911.02150.
7. Shazeer, *GLU Variants Improve Transformer*, arXiv:2002.05202.
8. Sun et al., *Retentive Network*, arXiv:2307.08621.
9. Ben Zaken et al., *BitFit*, ACL 2022.
10. Hu et al., *LoRA*, arXiv:2106.09685.
11. Zhang et al., *AdaLoRA*, ICLR 2023.
12. Houlsby et al., *Parameter-Efficient Transfer Learning for NLP*, ICML 2019.
13. Pfeiffer et al., *AdapterFusion*, EACL 2021.
14. Liu et al., *IA3 (T-Few)*, NeurIPS 2022.
15. Lester et al., *Prompt Tuning*, arXiv:2104.08691.
16. Li and Liang, *Prefix-Tuning*, arXiv:2101.00190.
