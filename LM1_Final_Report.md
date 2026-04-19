# Do Large Language Models Develop Dependency Grammar? Evidence from Attention-Derived Prefix Trees

**Authors:** Jani Ravi Kailash, Aditya Panwar, Ishan Trikha, Sandeep Kumar Gupta  
**Course:** Computational Linguistics (CSG) — Final Project  
**Date:** April 2026

---

## 1. Motivation for the Research Problem

Transformer-based Large Language Models (LLMs) such as BERT (Devlin et al., 2019) and GPT-2 (Radford et al., 2019) have achieved remarkable performance across virtually every NLP benchmark. Yet the internal representations that drive this performance remain only partially understood. A central question in interpretability research is whether these models learn the kind of *structural* knowledge that linguists consider fundamental to natural language — in particular, **dependency grammar**.

Dependency grammar (Tesnière, 1959; Mel'čuk, 1988) represents sentences as directed trees in which each word is linked to exactly one *head* word, capturing asymmetric syntactic relations (e.g., subject→verb, modifier→noun). Such representations are central to both formal linguistics and psycholinguistic models of human sentence processing (Gibson, 2000; Lewis & Vasishth, 2005).

Recent probing studies have shown that certain BERT attention heads track individual dependency relations with non-trivial accuracy (Clark et al., 2019), and that BERT's hidden representations encode parse-tree geometry recoverable by a linear probe (Hewitt & Manning, 2019). Voita et al. (2019) demonstrated that specific heads specialise for syntactic functions. However, these studies mainly evaluate *static* full-sentence performance. They do not address a more cognitively and computationally meaningful question: **how do the dependency structures derived from attention evolve as a sentence is built incrementally, token by token?**

Human language processing is fundamentally *incremental*: listeners and readers construct syntactic structure word by word, with each new word triggering only *bounded* structural updates (Nivre, 2004; Sturt & Lombardo, 2005). If an LLM's attention patterns encode genuine syntactic knowledge, we should expect a similar property — the dependency tree derived from attention should change only minimally when one token is appended.

**Research objective.** We investigate whether attention-derived dependency trees exhibit *structurally bounded updates* across sentence prefixes, and whether this stability is significantly greater than what random trees produce. We extend the analysis to four typologically diverse languages — English, Hindi, German, and French — to test the cross-lingual generality of any finding.

---

## 2. Hypotheses and Predictions

We formulate two testable hypotheses:

**H₁ (Structural Stability).** Dependency trees derived from transformer attention exhibit *bounded* incremental edge change as the sentence prefix grows. Formally, the Incremental Edge Change (IEC) metric decreases as prefix length increases, indicating that early structural commitments are largely preserved.

**H₂ (Non-Randomness).** The IEC values of attention-derived trees are *significantly lower* than those of independently generated random rooted trees, which serve as a null model with no structural memory.

**Predictions.** If H₁ and H₂ hold:

- **(P1)** LLM IEC curves will decrease monotonically or plateau as prefix length grows, while random baseline IEC remains high and approximately constant.  
- **(P2)** The middle layers (layers 5–8 of BERT-base) will show the lowest IEC, consistent with prior findings that these layers are most syntactically informative (Clark et al., 2019; Tenney et al., 2019).  
- **(P3)** The pattern will hold across all four languages, though IEC magnitudes may vary with typological properties (e.g., word-order freedom).

---

## 3. Methods

Our pipeline consists of three stages: data and attention extraction, tree construction, and stability evaluation.

### 3.1 Data

We draw sentences from four Universal Dependencies (UD) treebanks (Nivre et al., 2020):

| Language | Treebank     | Config  |  
|----------|-------------|---------|  
| English  | EWT         | en_ewt  |  
| Hindi    | HDTB        | hi_hdtb |  
| German   | GSD         | de_gsd  |  
| French   | GSD         | fr_gsd  |  

For each language we sample up to 100 sentences with word counts between 5 and 20. These treebanks provide gold-standard dependency annotations that we use as reference. Data is loaded by downloading CoNLL-U files directly from the official [Universal Dependencies GitHub repository](https://github.com/UniversalDependencies) and parsing them with the `conllu` Python library.

### 3.2 Attention Extraction

We use **bert-base-multilingual-cased** (Devlin et al., 2019) to ensure a single model processes all four languages under identical parameters. For a sentence of $n$ words $w_1, w_2, \ldots, w_n$, we construct prefix sequences $S_t = (w_1, \ldots, w_t)$ for $t = 2, 3, \ldots, n$.

Each prefix $S_t$ is tokenized into subword units and fed through the model. Let $A^{(l,h)} \in \mathbb{R}^{s \times s}$ denote the attention matrix at layer $l$ and head $h$, where $s$ is the subword sequence length. We aggregate across the $H$ attention heads within each layer by averaging:

$$\bar{A}^{(l)}_{i,j} \;=\; \frac{1}{H} \sum_{h=1}^{H} A^{(l,h)}_{i,j}$$

Because mBERT uses subword tokenization (WordPiece), we must **align subword attention to word-level attention**. For word $w_i$ mapped to subword set $\mathcal{S}_i$ and word $w_j$ mapped to $\mathcal{S}_j$:

$$A_{\text{word}}(w_i, w_j) \;=\; \frac{1}{|\mathcal{S}_i|\;|\mathcal{S}_j|} \sum_{s \in \mathcal{S}_i} \sum_{t \in \mathcal{S}_j} \bar{A}^{(l)}_{s,t}$$

This produces a $t \times t$ word-level attention matrix for each layer and each prefix.

### 3.3 Tree Construction — Chu-Liu / Edmonds Algorithm

We interpret the word-level attention as a weighted directed graph and extract the **maximum spanning arborescence** — the directed tree rooted at a virtual ROOT node that maximises total attention weight. This is the dependency tree that best reflects the collective attention pattern.

Formally, let $\mathcal{T}_{\text{ROOT}}$ denote the set of all directed spanning trees rooted at a virtual ROOT node. We solve:

$$T^{*} \;=\; \arg\max_{T \in \mathcal{T}_{\text{ROOT}}} \;\sum_{(j \to i) \in T} A_{\text{word}}(w_i, w_j)$$

where edge $j \to i$ means word $j$ is the *head* of word $i$. This optimisation is solved exactly in $O(n^2)$ time by the Chu-Liu/Edmonds algorithm (Chu & Liu, 1965; Edmonds, 1967), implemented via `networkx.minimum_spanning_arborescence` with negated weights.

### 3.4 Stability Evaluation

**Incremental Edge Change (IEC).** For prefix trees $T_{t-1}$ (length $t{-}1$) and $T_t$ (length $t$), IEC measures the fraction of the first $t{-}1$ words whose head changed:

$$\text{IEC}(t) \;=\; \frac{\bigl|\{i \in \{0,\ldots,t{-}2\} \;:\; \text{head}_{T_t}(i) \neq \text{head}_{T_{t-1}}(i)\}\bigr|}{t - 1}$$

IEC = 0 means perfect stability; IEC = 1 means every head reassigned.

**Tree Depth Change.** We also record the absolute change in tree depth:

$$\Delta d(t) \;=\; |d(T_t) - d(T_{t-1})|$$

**Random Baseline.** For each prefix length $t$, we independently generate a random recursive tree on $t$ nodes (each node $i \geq 1$ chooses its parent uniformly from $\{0, \ldots, i{-}1\}$) and compute IEC between consecutive random trees. We average over 200 Monte-Carlo trials. Because the random trees at lengths $t$ and $t{-}1$ are *independent*, this represents the maximum-volatility null hypothesis.

**Gold-Tree Reference.** We also compute IEC for gold UD trees restricted to each prefix. If a word's gold head falls outside the prefix, it is reattached to ROOT.

**Statistical Inference.** We compare LLM and random IEC distributions using a paired $t$-test and report Cohen's $d$ for effect size.

**Unlabeled Attachment Score (UAS).** As a sanity check, we compute UAS for the full-sentence attention-derived tree against the gold UD tree.

---

## 4. Results

We report results across all four languages using the middle layers (layers 5–8) of mBERT, which prior work identifies as the most syntactically informative.

### 4.1 Incremental Edge Change — LLM vs. Random Baseline (Figure 1)

![Figure 1: Structural Stability Curves](results/fig1_stability_curves.png)

The IEC curves reveal a clear separation between LLM attention trees and random baselines. For all four languages, the LLM IEC starts at approximately 0.32–0.40 for prefix length 3 and decreases steadily to 0.09–0.16 by prefix length 18–20. The random baseline, in contrast, remains roughly flat at 0.50–0.62 across all prefix lengths.

This confirms **H₁**: as the sentence grows, the attention-derived tree structure stabilises — earlier structural commitments are increasingly preserved. It also confirms **H₂**: the LLM stability is significantly greater than random ($p < 0.001$ for all languages; Cohen's $d$ ranging from 1.2 to 1.8, indicating large effect sizes).

Gold UD tree IEC values (English only) are the lowest, starting at ~0.15 and dropping to ~0.04, confirming that real dependency trees are inherently stable under prefixing and placing the LLM curves between the random baseline and the gold standard.

### 4.2 Cross-Lingual Comparison (Figure 2)

![Figure 2: Cross-Lingual Comparison](results/fig2_language_comparison.png)

Mean IEC values (averaged across all prefix lengths) are:

| Language | LLM Mean IEC | Random Mean IEC |  
|----------|-------------|----------------|  
| English  | 0.178       | 0.547          |  
| French   | 0.185       | 0.547          |  
| German   | 0.201       | 0.547          |  
| Hindi    | 0.228       | 0.547          |  

English and French (both SVO, relatively rigid order) show the highest stability. German (V2 order, freer constituent placement) is slightly less stable. Hindi (SOV, relatively free word order) shows the highest LLM IEC, likely reflecting its greater word-order flexibility and the prevalence of non-projective dependencies. Nevertheless, all four languages show IEC values *far* below the random baseline.

### 4.3 Layer-wise Analysis (Figure 3)

![Figure 3: Layer-wise Analysis](results/fig3_layer_analysis.png)

IEC varies substantially across BERT layers. Layers 1–3 (lowest) produce IEC values of 0.30–0.36, suggesting that early layers encode primarily positional or surface-level patterns that are unstable under prefix extension. Layers 5–8 achieve the minimum IEC (0.15–0.22), confirming **(P2)** that middle layers are the most syntactically structured. Layers 9–12 show a slight increase (0.22–0.28), consistent with the observation that upper layers shift toward more task-specific semantic representations (Tenney et al., 2019).

### 4.4 Tree Depth Volatility (Figure 4)

![Figure 4: Tree Depth Volatility](results/fig4_depth_change.png)

The mean absolute depth change per token added is 0.5–1.0 for LLM trees versus 1.5–2.5 for random trees. LLM trees exhibit bounded depth growth, consistent with the shallow dependency structures typical of natural language.

### 4.5 Unlabeled Attachment Score

Full-sentence UAS values (middle-layer average) range from 0.30 to 0.38 across languages. While modest compared to state-of-the-art parsers, these values are significantly above the random-tree UAS baseline (~0.12) and are consistent with prior attention-head probing results (Clark et al., 2019).

### 4.6 Summary

Both hypotheses are supported. LLM attention heads produce dependency-like structures that (a) stabilise as sentences grow, and (b) are far more stable than structureless random trees. The effect is robust across four typologically diverse languages and is most pronounced in BERT's middle layers.

---

## 5. Theoretical Implications

### 5.1 Implicit Syntactic Induction

Our results provide evidence that transformers, trained only on next-token prediction or masked-language modelling, develop representations that approximate dependency grammar. This is striking because no explicit syntactic supervision is provided. The bounded IEC we observe suggests that the model does not simply re-compute structure from scratch at each step; instead, it *incrementally refines* a persistent structural scaffold, much as a shift-reduce parser does.

### 5.2 Parallels with Human Incremental Processing

The decreasing IEC profile mirrors key properties of human sentence processing. Psycholinguistic models such as Surprisal Theory (Hale, 2001) and Dependency Locality Theory (Gibson, 2000) predict that processing difficulty — and by extension structural revision — is bounded by locality constraints. Our finding that LLM attention trees exhibit similar bounded revision is consistent with the hypothesis that efficient language processing, whether by humans or machines, converges on incremental, locally-bounded structural updates.

However, the mechanisms differ substantially. Human parsing is constrained by working-memory limitations and operates on a single serial stream, whereas BERT processes the entire prefix in parallel with no explicit memory bottleneck. The convergence in *behavioural* outcome (bounded revision) despite divergent *architectures* suggests that the constraint may arise from the statistical structure of language itself rather than from processing architecture per se.

### 5.3 Cross-Lingual Universality

The fact that the stability pattern holds across English, French, German, and Hindi — languages with differing word orders (SVO vs. SOV), morphological richness, and degrees of non-projectivity — suggests a language-universal tendency. This aligns with the typological observation that dependency length minimisation is a near-universal property of human languages (Futrell et al., 2015), and that neural language models recapitulate this pattern (Futrell & Levy, 2017).

### 5.4 Layer-wise Specialisation

The finding that middle layers are most stable reinforces the "syntactic middle" view of BERT's layer hierarchy (Jawahar et al., 2019; Tenney et al., 2019): lower layers encode surface-level features, middle layers capture syntactic structure, and upper layers encode more abstract semantic and task-relevant representations. This functional decomposition has implications for transfer learning, model pruning, and the design of syntax-aware models.

### 5.5 Limitations and Future Work

Our analysis uses head-averaged attention, which may dilute the contribution of attentionhead-specific syntactic roles. Future work could apply attention-head selection methods (Voita et al., 2019) or sparse attention extraction. Additionally, extending the analysis to autoregressive models (GPT-2, LLaMA) would test whether the stability property depends on bidirectional context. Finally, comparing our IEC metric with human reading-time data (e.g., eye-tracking corpora) could more directly test the cognitive plausibility of the bounded-revision hypothesis.

---

## References

- Chu, Y., & Liu, T. (1965). On the shortest arborescence of a directed graph. *Scientia Sinica*, 14, 1396–1400.
- Clark, K., Khandelwal, U., Levy, O., & Manning, C. D. (2019). What does BERT look at? An analysis of BERT's attention. *Proceedings of the 2019 ACL Workshop BlackboxNLP*.
- Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. *NAACL-HLT*.
- Edmonds, J. (1967). Optimum branchings. *Journal of Research of the National Bureau of Standards*, 71B, 233–240.
- Futrell, R., Mahowald, K., & Gibson, E. (2015). Large-scale evidence of dependency length minimization in 37 languages. *PNAS*, 112(33), 10336–10341.
- Futrell, R., & Levy, R. (2017). Noisy-context surprisal as a human sentence processing cost model. *EACL*.
- Gibson, E. (2000). The dependency locality theory: A distance-based theory of linguistic complexity. In A. Marantz et al. (Eds.), *Image, Language, Brain*, MIT Press.
- Hale, J. (2001). A probabilistic Earley parser as a psycholinguistic model. *NAACL*.
- Hewitt, J., & Manning, C. D. (2019). A structural probe for finding syntax in word representations. *NAACL-HLT*.
- Jawahar, G., Sagot, B., & Seddah, D. (2019). What does BERT learn about the structure of language? *ACL*.
- Lewis, R. L., & Vasishth, S. (2005). An activation-based model of sentence processing as skilled memory retrieval. *Cognitive Science*, 29(3), 375–419.
- Mel'čuk, I. (1988). *Dependency Syntax: Theory and Practice*. SUNY Press.
- Nivre, J. (2004). Incrementality in deterministic dependency parsing. *Workshop on Incremental Parsing*.
- Nivre, J., de Marneffe, M.-C., Ginter, F., et al. (2020). Universal Dependencies v2: An evergrowing multilingual treebank collection. *LREC*.
- Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). Language models are unsupervised multitask learners. *OpenAI Blog*.
- Sturt, P., & Lombardo, V. (2005). Processing coordinated structures: Incrementality and connectedness. *Cognitive Science*, 29(2), 291–305.
- Tenney, I., Das, D., & Pavlick, E. (2019). BERT rediscovers the classical NLP pipeline. *ACL*.
- Tesnière, L. (1959). *Éléments de syntaxe structurale*. Klincksieck.
- Voita, E., Talbot, D., Moiseev, F., Sennrich, R., & Titov, I. (2019). Analyzing multi-head self-attention: Specialized heads do the heavy lifting, the rest can be pruned. *ACL*.

---

## Appendix A: Complete Python Implementation

The complete, runnable pipeline is provided in the accompanying file `pipeline.py` and is publicly available on GitHub:

> **Repository:** [https://github.com/Sandeepgupta-24/LM1-dependency-grammar](https://github.com/Sandeepgupta-24/LM1-dependency-grammar)

The code implements all three stages described in §3 and generates the four figures referenced in §4. It is structured as follows:

1. **Data loading** (§§1–2 of the code): downloads UD treebanks as CoNLL-U files from the official Universal Dependencies GitHub repository and initialises `bert-base-multilingual-cased`.
2. **Attention extraction** (§§3–4): for each prefix of each sentence, extracts word-level attention matrices with subword-to-word alignment.
3. **Tree construction** (§5): builds maximum spanning arborescences using the Chu-Liu/Edmonds algorithm via `networkx`.
4. **Evaluation** (§§6–8): computes IEC, tree-depth change, UAS, gold-tree prefix IEC, and random baselines.
5. **Statistical testing** (§11): paired $t$-test with Cohen's $d$.
6. **Visualisation** (§12): four publication-quality figures using `matplotlib` and `seaborn`.

**Usage:**

```bash
git clone https://github.com/Sandeepgupta-24/LM1-dependency-grammar.git
cd LM1-dependency-grammar
pip install -r requirements.txt
python pipeline.py                     # Full run (100 sentences/language)
python pipeline.py --dry_run           # Quick test (5 sentences/language)
python pipeline.py --max_sentences 200 # Larger sample
```

**Dependencies:** Python ≥ 3.9, PyTorch ≥ 2.0, Transformers ≥ 4.30, conllu, networkx, numpy, matplotlib, seaborn, scipy (see `requirements.txt`).
