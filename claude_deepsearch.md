# Brutally Honest Academic Analysis of HPT‑QRC: Hybrid Photonic Temporal Quantum Reservoir Computing for Swaption Forecasting

## TL;DR

- **HPT‑QRC, in its current simulation‑only form, is not a Q1 journal paper.** A nearly identical preprint (Amanov & Azamov, *arXiv:2603.10707*, March 2026) already publishes the exact architectural recipe — an ensemble of fixed photonic reservoirs producing Fock‑basis features fed to a Ridge readout for swaption surface prediction — meaning the "novelty" the user thinks the project owns has already been pre‑empted in print and many of its core "contributions" must be reframed or dropped.
- **There is still a real, publishable paper here**, but it is a **methodology / benchmarking paper**, not a "we beat HAR with quantum" paper: the genuine novelty candidates are (i) the *temporal sliding‑window* version of HPT (vs. the single‑day surface model in 2603.10707), (ii) the *heterogeneous photon‑number ensemble* studied as an explicit ablation across NARMA‑10, Mackey‑Glass, S&P RV and VIX with HAR/HARX/RCX baselines and Diebold‑Mariano tests, and (iii) eventual *Quandela hardware* execution. None of these alone is Nature‑MI material; together, with hardware, they fit *Quantum Machine Intelligence*, *Quantum Science and Technology*, or *Physical Review Applied*, and as a simulation‑only paper they best fit a workshop (NeurIPS QML, QTML, ICML QML).
- **The biggest risks are academic‑integrity risks, not technical ones**: claims of "50× memory of classical ESN," "quantum advantage," "photonic," and "outperforms LSTM" are all unsafe as currently framed and would be flagged by any competent reviewer. The realistic path is: tone the claims down by ~70%, run the rigorous benchmark study described in §7–§8 below, get Quandela hardware results before submitting to a Q1 venue, and in the meantime submit a sharpened workshop paper to **QTML 2026** or the **NeurIPS 2026 Machine Learning and the Physical Sciences / QML workshop**.

---

## Key Findings

1. **Severe novelty collision with arXiv:2603.10707 (Amanov & Azamov, "Hybrid Photonic Quantum Reservoir Computing for High‑Dimensional Financial Surface Prediction," v1 11 Mar 2026).** This paper independently proposes essentially the same pipeline: an ensemble of three fixed photonic reservoirs ("QORC ensemble") producing Fock‑basis features, concatenated with a classical context vector, fed to a Ridge readout, applied to *swaption surface forecasting*, benchmarked against ~10 classical+quantum baselines. The authors' affiliation (QuanTech / New Uzbekistan University) and the timing (March 2026) suggest this group emerged from the *same EPFL / Quandela challenge ecosystem* that the user competed in. The headline architectural ideas the user lists — "Heterogeneous Photon Ensemble," fixed photonic reservoirs, Fock features, classical context concatenation, Ridge readout, 10 baselines on swaptions — are essentially **already in print**.

2. **The user's genuine, defensible incremental novelty is narrow**:
   - A *temporal* extension (sliding‑window over time series) rather than a single static surface compression. 2603.10707 is fundamentally a *spatial* compression+regression problem on 224‑dim surfaces; HPT‑QRC's `multi_qrc.py` `HPT_QRC_Multi` operates as a true time‑series reservoir.
   - Systematic *cross‑domain* benchmarking: NARMA‑10 + Mackey‑Glass + S&P 500 RV + VIX, rather than swaptions only. Most QRC papers test only chaos benchmarks; most volatility papers test only RV; doing both with the same model is uncommon.
   - HARX / ARMAX / RCX with exogenous variables as baselines on the *same* QRC architecture — this is methodologically stronger than 2505.13933 and 2603.10707, both of which use weaker baselines.
   - Diebold‑Mariano on MSE *and* QLIKE — standard in econometrics (Patton & Sheppard 2009 established QLIKE as having highest DM power), almost never seen in QRC papers, which is a defensible contribution to *evaluation rigor*.

3. **The "50× memory capacity over classical ESN" claim is unsafe.** Memory capacity (MC) for ESNs is bounded by the number of linearly independent reservoir nodes (Jaeger 2002); a classical ESN of comparable readout dimensionality (~1,200 features) has MC bounded by ~N. Reporting "50×" almost certainly reflects an unfair comparison: small classical ESN (e.g., 50 nodes) vs. a photonic reservoir whose Fock‑basis output dimensionality is in the hundreds to thousands. This is a textbook reviewer rejection trigger. The literature (Suzuki et al. 2022, Molteni/Prati 2023, Kora et al. 2024 on entanglement‑MC) is explicit that QRC memory advantages are *modest and conditional*, not order‑of‑magnitude.

4. **"Photonic" is misleading until Quandela hardware is run.** What HPT‑QRC actually computes is a *classical simulation of linear‑optical Fock‑state probabilities via Perceval/MerLin's Strong Linear Optical Simulation (SLOS)*. This is mathematically equivalent to evaluating permanents/sub‑permanents — it is *not* photonic computing in any operational sense. The genuine "quantum/photonic" property (boson‑sampling #P‑hardness, which 2603.10707 invokes) only matters when (a) the reservoir is large enough that simulation is intractable, or (b) it is run on real hardware. Neither holds for the user's `photon_list=[2,3,4]` with small mode counts; classical simulation in milliseconds proves this. This must be flagged transparently.

5. **The "outperforms LSTM" framing is fragile** because (a) on small datasets like swaption surfaces or daily RV (~thousands of samples), Ridge regression on rich nonlinear features routinely beats LSTM regardless of the feature source — Branco et al. 2024 ("HARd to Beat") explicitly show HAR beats LSTM on RV; (b) LSTM hyperparameter tuning matters enormously; (c) the architectural advantage shown is therefore one of *regularization‑under‑small‑data*, not of quantumness. 2603.10707 makes this point honestly ("Regularised linear readouts outperform deep learning when data is limited") — the user should follow suit.

6. **The three reference papers, properly characterized**:
   - **2603.10707 (Amanov & Azamov, March 2026):** *Direct competitor / near‑duplicate.* Almost the same architecture, same domain (swaptions), same QORC formulation (citing the QORC paper of Pauly/Bautista et al. and the Photon‑QuaRC framework of Nerenberg et al. 2024). Must be cited as concurrent independent work, distinguished on temporal axis and benchmarks. Cannot claim "first" anything they have already claimed.
   - **2505.13933 (Li, Mukhopadhyay, Bayat, Habibnia, May 2025; v2 April 2026):** *Methodologically the most relevant prior work for the volatility experiments.* Uses a transverse‑field Ising QRC for *realized volatility forecasting* with HAR and ML baselines. Different reservoir physics (qubit Ising spins vs. linear‑optical Fock) but same target task; the user must cite, replicate, and beat (or honestly tie with) this paper to claim anything in volatility forecasting.
   - **2510.25183v1 (Kodali et al., Oct 2025), "Sustainable NARMA‑10 Benchmarking for QRC":** A *narrow benchmarking paper* comparing QRC vs. ESN/LSTM/QLSTM on NARMA‑10 with a "sustainability" (compute cost) angle. Useful for benchmark methodology and as a citation, but not architecturally close. The user's NARMA‑10 study should explicitly position against this.

7. **Paper class verdict.** Currently the project sits between (a) a strong workshop paper and (b) a weak journal paper. With Quandela hardware execution it can credibly target *Quantum Machine Intelligence* (Springer), *Quantum Science and Technology* (IOP), or *Physical Review Applied*. Without hardware, *Q1 is unrealistic* once 2603.10707 is in the citation graph.

---

## Details

### 1. Project understanding (technical)

Working from the user's description, the public Amanov/Azamov preprint (which uses the same nomenclature and pipeline as the EPFL/Quandela hackathon Quandela Challenge), the LinkedIn post confirming "HPT‑QRC = Hybrid Photonic Temporal QRC, MerLin‑powered photonic quantum reservoir, EPFL Quantum Hackathon 2026 winner," and the technical components named in the task:

- **What is *quantum*:** Mathematical objects — Fock states, linear‑optical unitaries acting on multi‑photon states, output probabilities computed via Perceval's SLOS backend (which evaluates permanents of submatrices). This is genuinely quantum *physics* but executed as a deterministic classical simulation.
- **What is *photonic*:** Currently *only the model*. The Mach–Zehnder/beamsplitter unitary and Fock‑basis measurement vocabulary are photonic; the *execution* is on CPU/GPU. Calling the simulated artifact "photonic" is acceptable in the literature only if accompanied by a clear "simulation" disclaimer.
- **What is *classical*:** (i) The temporal sliding window construction; (ii) optional sparse denoising autoencoder front‑end (in 2603.10707); (iii) the Ridge readout (closed‑form OLS with L2); (iv) all baselines (AR1/AR3, HAR, HARX, ARMAX, LSTM, ESN, RCX). Note that ESN is itself a *classical reservoir* — it's the quantum reservoir's intended apples‑to‑apples comparator.
- **What is *financial*:** Swaption volatility surfaces (originally), S&P 500 realized variance/volatility (extension), VIX (extension). HARX/ARMAX/RCX exogenous regressors are standard volatility econometrics.
- **What is the "Heterogeneous Photon Ensemble":** Three (or more) fixed linear‑optical reservoirs run with different photon numbers (e.g., n=2, 3, 4 in the same number of modes). Their Fock‑basis output features are concatenated. Each photon‑count produces a different Fock‑space dimensionality and different statistics of multi‑photon interference. The user must be aware: (i) 2603.10707 already does this with three reservoirs producing 1,215 features; (ii) Nerenberg et al. 2025 ("Photon‑QuaRC," Optica) systematically vary photon number and encoding in a single reservoir; (iii) the diverse‑timescale / multi‑reservoir ESN literature (Tanaka et al., DTS‑ESN PRR 2022; HP‑MRESN 2023) already covers heterogeneous *classical* reservoir ensembles. So this is an *application of a known multi‑reservoir idea to the photon‑number axis*, not a new architectural class.
- **What is the "Memory Capacity 50× ESN" claim:** Almost certainly the standard Jaeger linear MC computed on the photonic features versus a small ESN. Until apples‑to‑apples (matched feature dimension, matched hyperparameter sweep) is done, the number is meaningless and dangerous.
- **`HPT_QRC_Multi` (multi_qrc.py):** From the description, an ensemble class that constructs `len(photon_list)` photonic reservoirs at different photon counts, concatenates their Fock features over a temporal sliding window, and provides them to a downstream readout (likely Ridge). This is a clean, honest engineering artifact — the right contribution to *open‑source* alongside the paper.

### 2. Related‑work comparison (deep)

**arXiv:2603.10707 — Amanov & Azamov, "Hybrid Photonic Quantum Reservoir Computing for High‑Dimensional Financial Surface Prediction" (March 2026).**
- *Pipeline:* sparse denoising autoencoder (224→20 latent) → 1,215 Fock features from three fixed photonic reservoirs → concatenate with 120‑dim classical context → Ridge → reconstruct surface.
- *Domain:* Swaption surface, six held‑out trading days.
- *Baselines:* 10 classical+quantum, including VQC, Quantum LSTM (which yield negative R²), classical Ridge, deep models.
- *Headline result:* lowest surface RMSE = 0.0425, sub‑millisecond inference.
- *Theoretical framing:* identical to what the user described — boson sampling #P‑hardness, fixed quantum dynamics avoid barren plateaus, training is convex.
- *Overlap with HPT‑QRC:* Architecture, target domain, conceptual contribution, framing — all strongly overlap. Both use Perceval‑style linear optics; both use ensembles of fixed photonic reservoirs; both concatenate quantum and classical features; both use Ridge readout; both report "outperforms variational quantum and deep learning baselines."
- *Differences (and the user's defensible space):*
  1. 2603.10707 treats each day as a *static high‑dim surface* prediction. HPT‑QRC's `HPT_QRC_Multi` is *temporal*, with a sliding window — this is a real algorithmic difference.
  2. 2603.10707 evaluates only on swaptions; HPT‑QRC adds NARMA‑10, Mackey‑Glass, S&P RV, and VIX.
  3. 2603.10707 uses six held‑out days (very small test set, a weakness reviewers will note); HPT‑QRC has more capacity to do longer rolling/expanding windows.
  4. 2603.10707 does not (visibly) report Diebold‑Mariano or QLIKE; HPT‑QRC does — this is a substantive evaluation upgrade.
  5. HPT‑QRC plans Quandela hardware execution; 2603.10707 is simulation‑only.
- *What MUST be cited:* this paper, prominently, in introduction and related work, with text along the lines of *"Concurrent independent work by Amanov & Azamov (2026) explores a closely related architecture for static swaption surface reconstruction; we differ in (i) the temporal sliding‑window formulation, (ii) systematic cross‑domain benchmarking on NARMA‑10, Mackey‑Glass, S&P 500 RV and VIX, (iii) econometric evaluation via Diebold‑Mariano on MSE and QLIKE, and (iv) hardware execution on the Quandela Ascella/Belenos platform."*
- *What MUST NOT be claimed:* "first hybrid photonic QRC for financial forecasting," "first ensemble of fixed photonic reservoirs," "first to use Fock‑basis features with classical context for swaptions," "first to combine fixed photonic reservoirs with Ridge," "first to compare with VQC/Quantum LSTM in finance."

**arXiv:2505.13933 — Li, Mukhopadhyay, Bayat, Habibnia, "Quantum Reservoir Computing for Realized Volatility Forecasting" (May 2025; v2 April 2026).**
- *Reservoir:* fully connected transverse‑field Ising Hamiltonian with input/memory qubit separation. Qubit‑based, *not* photonic.
- *Target:* realized volatility (the same task as HPT‑QRC's S&P RV extension).
- *Baselines:* econometric (HAR family) and standard ML.
- *Why it matters:* this is the *direct prior on the volatility task*. The user must (a) cite it, (b) ideally reproduce a comparable Ising‑QRC baseline on the same data split, (c) frame their photonic reservoir as a *complementary platform*, not a replacement, since 2505.13933 has the publication priority on the volatility application. Failure to cite or beat this paper on its own ground will be flagged.
- *Distinction:* photonic platform (boson sampling vs. Ising spin); higher feature dimensionality at low photon counts than equivalent qubit counts; realistic hardware path (Quandela vs. arbitrary qubit Ising platform).

**arXiv:2510.25183v1 — Kodali, Singh, et al., "Sustainable NARMA‑10 Benchmarking for QRC" (Oct 2025).**
- A focused benchmarking paper: ESN vs. LSTM vs. QLSTM vs. (qubit) QRC on NARMA‑10, with computational cost and "sustainability" angle, plus memory capacity reporting.
- *Useful for:* (a) an evaluation template for the user's NARMA‑10 study; (b) a citation for the legitimacy of the QRC + ESN + LSTM benchmark axis; (c) a counter‑example that QRC memory capacity advantages are real but moderate (their MC is competitive, not 50×).
- *Distinction for HPT‑QRC:* photonic vs. qubit QRC, multi‑task (not just NARMA‑10), and adding HAR‑family baselines.

### 3. Broader literature landscape (concise survey)

- **Photonic QRC (2023–2026).** García‑Beni et al. (Phys. Rev. Applied 2023, *Scalable photonic platform*); Nerenberg et al., *Photon‑QuaRC* (Optica 2025) — most directly methodologically aligned, photon‑number resolving QRC with Perceval; Cimini et al. *Large‑scale GBS reservoir* (Phys. Rev. X / arXiv 2505.13695, 2025) — Gaussian boson sampler with >400 modes, MNIST/spoken vowels; Ekici 2026 (*Programmable linear‑optical QRC with measurement feedback*, arXiv:2602.17440) — closely related architecture with measurement feedback; Carles et al. 2026 (*Experimental QRC with circuit QED*); Nature Photonics 2026 (*Experimental memory control in CV optical QRC*); Bienstman group / Ghent (passive cavity, NALM 2025 *Nat. Commun.*); Vinckier et al. coherently driven cavity. Also Kar & Babu 2025 (*HPQRC*, arXiv:2511.09218) — note: this paper exists, but its "27%/35% improvements" are simulated and weakly grounded; cite cautiously.
- **QRC for finance.** Li et al. 2505.13933 (Ising RV); Amanov & Azamov 2603.10707 (photonic swaption); a small literature on quantum methods for option pricing (mostly QC/Monte‑Carlo, not RC). Realized‑volatility ML baselines: Branco et al. 2024 ("HARd to Beat"); Patton & Sheppard 2009 (QLIKE/DM theory); foundation‑model RV (Goel et al. 2025); GNAR‑HARX (2510.24443).
- **Heterogeneous reservoirs.** DTS‑ESN (PRR 2022); Mod‑Deep ESN; HP‑MRESN (2023); LS‑CrossESN; *Topological entropy and multiplexed ensembles* (Halder et al. 2026, bioRxiv). The "ensemble of reservoirs at different scales" idea is mature; the specific axis "different photon numbers" is the user's narrow novelty.
- **Memory capacity.** Jaeger 2002 (canonical MC definition); Suzuki et al. (Sci. Rep. 2022, natural QRC); Molteni & Prati (2023, memory reset rate); Kora et al. (2024, frequency/dissipation entanglement advantage); ESP extensions (arXiv:2403.02686).
- **Boson sampling for ML.** Pauly/Bautista QORC (Optica Quantum 2025); Cimini et al. (GBS reservoir, 2025); Photon‑QuaRC (Nerenberg 2024–2025); the *Photonic Quantum‑Accelerated ML* paper (arXiv:2512.08318).
- **Quandela hardware.** Ascella (Maring et al., *Nat. Photonics* 2024, "A versatile single‑photon‑based quantum computing platform," 12 modes, 6 photons, 4 Hz sampling); Belenos (12 qubits, second‑gen); MosaiQ (6–24 qubits); Perceval (Heurtel et al., *Quantum* 2023); MerLin 0.3 (PyTorch‑native, differentiable photonic QML, 2025). This hardware envelope is *exactly the regime* where small Fock‑state QRC fits.

### 4. Novelty assessment (rigorous)

| Type | Verdict | Notes |
|---|---|---|
| **Engineering** | Moderate | Clean, modular, reproducible HPT_QRC_Multi class with NARMA10/MG/RV/VIX runners. Open‑sourcing this is a real contribution. |
| **Application** | Weak‑to‑moderate | Swaption forecasting was the hackathon scope; 2603.10707 already publishes this. RV/VIX with photonic QRC is fresh but Ising‑QRC (2505.13933) precedes. |
| **Methodological (heterogeneous photon ensemble)** | Weak novelty | Multi‑reservoir ensembles are well‑known classically (DTS‑ESN, MRESN); photon‑number ensembling is a *new axis* but its theoretical motivation is shallow without a memory‑nonlinearity tradeoff analysis. Could become moderate novelty if you *prove* (e.g., empirically, with controlled experiments) that varying photon number provides a different memory–nonlinearity balance than varying mode count or beam‑splitter randomness alone. |
| **Theoretical** | None as currently described | No new theorem, no new universality result, no closed‑form MC bound. |
| **Experimental (sim only)** | Low | Simulation studies of Fock‑state RC are common. |
| **Experimental (with Quandela hardware)** | Moderate‑to‑strong | Running real photonic QRC on Ascella/Belenos for a financial time‑series task would be one of the *first* such hardware demonstrations and is the strongest possible upgrade path. |

**Safe** novelty claims: methodology benchmarking; econometric evaluation of QRC; cross‑domain reservoir ensembling.
**Risky:** "first photonic QRC for finance," "quantum advantage," "50× memory."
**Reject‑bait:** "outperforms classical," "state‑of‑the‑art," "novel architecture" without ablation that isolates the photon‑number axis from the readout.

### 5. Academic integrity analysis

- **Overclaims to remove or soften:**
  - "50× better memory than classical ESN" → "Compared with a *matched‑feature‑dimension* ESN baseline, HPT‑QRC achieves ~X% higher linear memory capacity at fixed readout dimension on the standard Jaeger MC benchmark; we explicitly avoid claims of order‑of‑magnitude advantage as small classical reservoirs would underrepresent ESN's known scaling."
  - "Photonic" (in title and abstract) → keep "photonic" only if you commit to (a) presenting the Quandela hardware run, or (b) using "linear‑optical (simulated)" or "photonic‑model" consistently in abstract/methods. The accepted norm in the QRC community is to call it "simulated photonic QRC" until run on hardware.
  - "Quantum advantage" → never use this phrase. Use "quantum/photonic feature extractor" and discuss expressivity, not advantage.
  - "Outperforms LSTM" → "On small‑sample volatility data, the photonic ensemble + Ridge readout matches or outperforms an LSTM baseline of comparable parameter budget; we attribute this primarily to the convex readout regularization regime where deep models underperform on limited data (Branco et al. 2024)."
- **Ambiguous terminology to fix:** "temporal QRC" (too generic), "hybrid" (used 4 different ways in the literature; specify *quantum‑classical feature concatenation* or *photonic + classical readout*), "ensemble" (specify whether it's a *parallel multi‑reservoir* or a *bagging ensemble*; from the description it's parallel multi‑reservoir, which is more honest).
- **Citation gaps to close:** 2603.10707; 2505.13933; Nerenberg 2024/2025 (Photon‑QuaRC); García‑Beni 2023; Cimini 2025 (GBS); Carles 2026 (cQED experimental QRC); Maring 2024 (Ascella *Nat. Photonics*); Heurtel 2023 (Perceval, *Quantum*); Pauly et al. 2025 (QORC, Optica Quantum); Patton & Sheppard 2009 (QLIKE/DM theory); Corsi 2009 (HAR); Branco et al. 2024 ("HARd to Beat"); Tanaka et al. 2019 (physical RC review); Jaeger 2002 (MC definition).
- **Reproducibility gaps:** seeds for SLOS sampling; train/val/test split definitions for RV (calendar split? Rolling window?); whether VIX is used as exogenous (HARX‑style) or target; hyperparameter search protocol for Ridge α and ESN size; *what counts as one parameter* in parameter‑count comparisons (you cannot count fixed reservoir parameters for QRC and trainable LSTM weights symmetrically).
- **Dataset leakage risks:** (a) Standardization/scaling computed on full series rather than train‑only — *very common bug*; (b) using forward‑looking features (VIX at time t to predict RV at time t — VIX is implied vol of next 30 days, must lag); (c) for HARX, ensure all daily, weekly, monthly aggregates are right‑aligned to t‑1; (d) walk‑forward vs. fixed split — pre‑declare and stick.
- **Weak baselines:**
  - The classical Ridge baseline must use the *same temporal window features* as HPT‑QRC, otherwise the comparison is feature‑engineering vs. quantum, not classical vs. quantum.
  - ESN size must be tuned (grid search reservoir size 50–2000, leak rate, spectral radius) — most QRC papers cherry‑pick a small ESN.
  - LSTM must have early stopping, dropout, and architecture sweep; otherwise you are comparing a tuned linear model with an untuned RNN.
  - Add a *Random Fourier Features + Ridge* baseline — this is the *non‑quantum analog* of "random fixed nonlinear features + Ridge" and is the toughest comparator. If HPT‑QRC does not beat RFF+Ridge of matched feature dimension on RV, the quantumness claim collapses.
- **Cherry‑picking risks:** reporting only the best photon_list configuration; reporting the best of N seeds; reporting only one volatility regime; one swaption tenor. Pre‑register or provide all seeds and all configurations in supplementary.

### 6. Technical weaknesses (specific)

- **Reservoir architecture.** No fading‑memory / echo‑state property proof for the multi‑photon ensemble; no analysis of the *memory–nonlinearity tradeoff* (Dambre et al. 2012 framework). Fix: compute IPC (information processing capacity) curves per photon number.
- **Temporal encoding.** Sliding window injects past inputs as features but does not give the reservoir true recurrence — without a feedback loop or measurement‑conditioned phase update (Ekici 2026), the system is closer to a *quantum kernel feature map over windows* than to a true RC. Be transparent: "sliding‑window quantum extreme learning machine variant" might be a more accurate label than "reservoir."
- **Photonic simulation.** SLOS scales factorially; verify mode and photon counts are tractable. Document precise (n, m) settings and runtime. Add shot noise simulation (finite‑sample Fock probabilities) — without it, the simulated‑vs‑hardware gap is huge.
- **Quantum feature extraction.** The map x → P(Fock outputs | input‑modulated unitary) — specify *exactly* how time‑series values modulate phases (phase encoding amplitude, range, periodicity). Encoding normalization is the single most important hyperparameter and is rarely reported.
- **Classical readout.** Ridge α should be selected by walk‑forward CV, not single train/val.
- **Preprocessing.** Robust scaler for fat‑tailed financial returns is correct; document inverse‑transform exactly; for log‑RV, train in log space and back‑transform with bias correction (Patton).
- **Train/test split.** Six held‑out days (as in 2603.10707) is far too small. For RV, use ~20% rolling held‑out (e.g., last 250–500 days for a 5‑year sample) with a moving window.
- **Evaluation metrics.** RMSE alone is insufficient for volatility — QLIKE has higher DM power (Patton & Sheppard 2009). MAE for robustness. R²_OS (Campbell‑Thompson out‑of‑sample R²). Coverage of volatility VaR for a downstream utility check.
- **Statistical significance.** DM with Newey‑West HAC variance, h=horizon‑specific. Add Hansen MCS test (Model Confidence Set) to account for multiple comparisons across the >10 baselines — simple DM is multiple‑testing biased.
- **Hyperparameter tuning.** Equal compute budget per model (e.g., 100 trials of Optuna). Pre‑register search spaces.
- **Computational cost.** Honest accounting: include simulator wall‑clock and memory; for hardware, include shot count and time per inference.

### 7. Research roadmap (three tiers)

**A. Workshop‑grade (achievable in 4–6 weeks, simulation‑only).**
- Drop swaptions to secondary; lead with NARMA‑10, Mackey‑Glass, S&P 500 RV.
- Single‑task ablations on photon_list ∈ {[2], [3], [4], [2,3], [2,3,4]}, with matched feature dimension.
- Compare against ESN, RFF+Ridge, HAR, HARX, LSTM with proper tuning.
- DM+QLIKE on RV, NRMSE on chaotic benchmarks.
- Memory‑capacity / IPC analysis with Dambre framework.
- *Target venues:* QTML 2026; NeurIPS 2026 *Machine Learning and the Physical Sciences* workshop; NeurIPS 2026 *Quantum Machine Learning* workshop; ICML 2026 QML workshop.

**B. Strong version (3–4 months, simulation + small Quandela run).**
- Adds: 6–12 photon, 12‑mode runs on Quandela Ascella/Belenos via Perceval cloud; comparison of simulated vs. measured Fock probabilities on a subset of inputs; shot‑noise robustness; ablation isolating the photon‑number ensemble axis vs. mode‑count axis.
- Memory–nonlinearity tradeoff figure (IPC plane) per photon configuration.
- Walk‑forward RV evaluation across 2010–2024 with regime analysis.
- *Target venues:* *Quantum Machine Intelligence* (Springer, Q1‑adjacent), *Quantum Science and Technology* (IOP, Q1), *Physical Review Applied* (Q1), *Machine Learning: Science and Technology* (Q2), *Quantitative Finance* (for the financial framing).

**C. Ambitious (9–12 months, full hardware story).**
- Multi‑hour Quandela hardware run on RV across hundreds of days; demonstrate *practical* low‑latency inference (sub‑second) on real photonic hardware end‑to‑end.
- Theoretical contribution: a closed‑form or semi‑analytic argument for why photon‑number ensembling expands the IPC envelope (or admit that classical multi‑reservoir matches it and frame photonic version as a hardware‑native instantiation).
- Compare with Gaussian boson sampler RC (Cimini 2025) at matched feature dimension.
- *Target venues:* npj Quantum Information (top of QML hierarchy), Nature Communications (if the hardware story is unprecedented), PRX Quantum (theory + hardware combo), Optica Quantum (photonics‑angled). Nature Machine Intelligence is *unrealistic* without a substantially more general algorithmic claim.

### 8. Experimental plan

**Research questions:**
- RQ1. Does a photon‑number heterogeneous photonic ensemble produce a measurably richer feature map than a single‑photon‑count photonic reservoir of *matched* total feature dimension?
- RQ2. Does HPT‑QRC + Ridge match HAR/HARX on out‑of‑sample QLIKE for S&P 500 RV across calm and turbulent regimes?
- RQ3. How does simulated performance translate to Quandela hardware execution under realistic shot counts and indistinguishability?
- RQ4. Does the photonic ensemble beat Random Fourier Features + Ridge of matched feature dimension on chaotic and financial benchmarks?

**Datasets:**
- NARMA‑10 (standard generation, fixed seeds).
- Mackey‑Glass (τ=17), 10 seeds.
- Realized Library 5‑min RV for SPX 2010–2024.
- VIX as exogenous regressor.
- Swaption surfaces (replicating 2603.10707's setup) for cross‑comparison.

**Baselines:** AR(1), AR(3), HAR, HARX, ARMAX, LSTM (1‑3 layers, tuned), GRU, ESN (sizes 100/500/2000), Random Fourier Features + Ridge, classical Ridge on temporal window, Quantum LSTM (small), Ising‑QRC reproduction of 2505.13933 (if feasible), single‑photon photonic reservoir (n=2 only), photonic reservoir at n=4 only.

**Ablations:** photon_list combinations; mode count; encoding amplitude; ensemble vs. single; quantum features only vs. classical context only vs. concatenated; effect of shot noise (102, 103, 104, 105 shots); effect of photon distinguishability (0%, 10%, 30%).

**Metrics:** RMSE, MAE, QLIKE, R²_OS, NRMSE (chaos), Diebold‑Mariano t‑stats with HAC, Hansen MCS p‑values, IPC (linear + nonlinear capacities).

**Figures (publication‑ready):** (1) pipeline diagram; (2) IPC plane vs. photon configuration; (3) walk‑forward QLIKE bar chart with DM significance brackets; (4) MCS bubble plot of model survivors; (5) sim‑vs‑hardware Fock‑probability scatter; (6) latency vs. accuracy Pareto frontier.

**Tables:** per‑model NRMSE on each benchmark; per‑model QLIKE on RV split into calm/turbulent; ablation table; computational cost table.

**Expected reviewer concerns:** (a) overlap with 2603.10707 — pre‑empt by explicit positioning; (b) why photonic vs. qubit — answer with hardware accessibility (Quandela cloud) and feature‑dimension scaling; (c) where is the quantum advantage — answer honestly: this is *not* a quantum‑advantage paper, it is an *expressivity / hardware‑native* paper; (d) shot noise — answer with explicit ablation; (e) baselines — answer with RFF+Ridge being beaten or honestly tied with explanation.

### 9. Paper structure

1. **Title** (see §10).
2. **Abstract** — state the architecture in one sentence; state the *concurrent* nature of 2603.10707 in the introduction (not abstract); list datasets; headline metric + DM significance.
3. **Introduction** — temporal forecasting with QRC; gap = photon‑number heterogeneity + econometric evaluation rigor; contributions list (3, conservative).
4. **Background and related work** — RC + QRC + photonic QRC + RV forecasting + multi‑reservoir RC. Explicit "concurrent and independent work" paragraph for 2603.10707.
5. **Method** — sliding window encoding; per‑photon photonic reservoir formalism (Fock state evolution under linear‑optical unitary); ensemble concatenation; classical context; Ridge readout. Pseudocode + computational complexity.
6. **Experimental setup** — datasets, splits, baselines, hyperparameter search, hardware/simulator details, shot noise modeling.
7. **Results** — ordered by RQ1‑RQ4. Memory/IPC, then chaos, then RV/VIX. DM and MCS tables.
8. **Hardware execution (Quandela)** — separate section if hardware results are obtained; otherwise a "Toward hardware" section discussing realistic shot/photon constraints.
9. **Discussion** — what the ensemble axis adds; honest limitations (shot noise, scaling with photon count, no quantum advantage shown); when the model fails (turbulent regimes, regime breaks).
10. **Conclusion** — one paragraph; do not oversell.
11. **Reproducibility** — full code, seeds, configs, dataset preprocessing notebook.
12. **Appendix** — extra ablations, all DM matrices, encoding sensitivity.

### 10. Suggested titles, contribution statements, claims

**Five academically appropriate titles:**
1. *Heterogeneous Photon‑Number Ensembles in Linear‑Optical Quantum Reservoir Computing for Financial and Chaotic Time‑Series Forecasting*
2. *A Temporal Photonic Quantum Reservoir for Realized Volatility: Benchmarks, Memory Capacity, and Diebold–Mariano Evaluation*
3. *Multi‑Photon Reservoir Ensembles: Photon‑Number Heterogeneity as a Computational Resource for Time‑Series Forecasting*
4. *From Simulation to Quandela Hardware: A Photon‑Ensemble Quantum Reservoir for Volatility and Chaotic Benchmarks*
5. *Photonic Quantum Reservoirs at Different Photon Numbers: A Cross‑Domain Benchmarking Study Against Classical and Econometric Baselines*

**Three conservative contribution statements (recommended):**
- We introduce *HPT‑QRC*, a temporal sliding‑window photonic quantum reservoir whose feature space is built by concatenating Fock‑basis outputs of multiple fixed linear‑optical reservoirs with different photon numbers, and study its behavior on NARMA‑10, Mackey‑Glass, S&P 500 realized volatility, and VIX‑augmented forecasts.
- We provide the first econometrically rigorous evaluation of a photonic QRC on realized volatility, including QLIKE, Diebold–Mariano, and Model Confidence Set tests against the HAR/HARX and ESN/LSTM/RFF+Ridge baseline families.
- We characterize the simulated‑to‑hardware gap by running matched configurations on Quandela's photonic platform (Ascella/Belenos/MerLin) and reporting shot‑noise and indistinguishability sensitivities.

**Three stronger (only if backed by hardware + matched ablation):**
- The photon‑number ensembling axis expands the information‑processing capacity envelope beyond what is achievable by varying mode count or beam‑splitter randomness alone at matched feature dimension.
- Photonic feature maps with Ridge readout match HAR‑family baselines on QLIKE for S&P 500 RV out‑of‑sample (DM p > 0.10 in MCS), establishing parity rather than dominance — a meaningful result given the strength of HAR.
- We demonstrate end‑to‑end realized‑volatility inference on Quandela Ascella photonic hardware at sub‑second latency, the first such hardware demonstration in financial econometrics.

**Three claims to AVOID:**
- "First photonic QRC for finance" (2603.10707 precedes; 2505.13933 precedes for QRC + RV).
- "Quantum advantage" of any kind.
- "50× memory of classical ESN" or any order‑of‑magnitude claim against ESN without matched‑dimension comparison.

### 11. Reviewer simulation

| # | Likely criticism | Question | Evidence needed | If unanswered: |
|---|---|---|---|---|
| 1 | "This duplicates Amanov & Azamov 2603.10707." | "What does HPT‑QRC do that 2603.10707 doesn't?" | Side‑by‑side method comparison; explicit temporal extension; cross‑domain benchmarks; DM/QLIKE rigor; hardware run. | **Reject.** |
| 2 | "Photonic vs. simulated photonic — which is it?" | "Have you executed on hardware?" | Quandela Ascella/Belenos run with concordance plot. | **Major revision** without; *Reject as photonic* if title still says "photonic" but only simulation. |
| 3 | "ESN baseline is too weak; 50× MC claim is fishy." | "Provide MC scaling vs. ESN size, matched feature dim." | Sweep ESN N=50…2000; report MC vs. N curves; remove order‑of‑magnitude claim. | **Major revision.** |
| 4 | "HAR is hard to beat on RV (Branco 2024)." | "Do you beat HAR/HARX on QLIKE with DM significance?" | DM table + MCS; honest report even if you only tie. | **Major revision** (tying is acceptable if honest). |
| 5 | "What is the quantum contribution? Random Fourier features would do this." | "Run RFF+Ridge of matched feature dimension." | RFF+Ridge baseline; analysis of where quantum features win/lose. | **Reject** for top venues; **revise** for workshops. |

**Likely verdict by venue:**
- Top journal (Nature MI, npj QI, PRX): **Reject** without hardware + RFF+Ridge baseline.
- Mid‑tier journal (QMI, QST, Phys. Rev. Applied): **Major revision** with hardware; possibly accept after revisions.
- Workshop (QTML, NeurIPS QML, ML4PS): **Accept** with current scope after toning down claims.

### 12. Final verdict

- **Worth publishing? Yes — but not as a Q1 journal paper today.** As of May 2026 with 2603.10707 already on arXiv (March 2026), the headline architectural claim is gone; what's left is a sound *benchmarking + econometric‑rigor + temporal‑extension* paper that is workshop‑strong and journal‑ready only after Quandela hardware execution.
- **Realistic publication path (named venues, in priority order):**
  1. **QTML 2026** (international conference on Quantum Techniques in Machine Learning) — strong fit; community appreciates honest benchmarks.
  2. **NeurIPS 2026 *Machine Learning and the Physical Sciences* workshop** — excellent for cross‑disciplinary photonic + finance positioning.
  3. **NeurIPS 2026 / ICML 2026 QML workshop** — good fit for the simulation‑only version.
  4. **Quantum Machine Intelligence** (Springer; reasonable Q1‑adjacent target after hardware run).
  5. **Quantum Science and Technology** (IOP, Q1) — strong fit if the IPC + memory‑capacity analysis is rigorous.
  6. **Physical Review Applied** (Q1) — fit if hardware demonstration is included.
  7. **Quantitative Finance** or **Journal of Financial Econometrics** — for a finance‑first reframing emphasizing the QLIKE/DM/MCS rigor; novelty there is "photonic feature engineering for RV," and these venues are friendlier to non‑quantum‑advantage claims.
  8. **Quantum (open‑access)** — if the paper becomes more theoretical (IPC bounds, ESP proofs).
  9. *Avoid* Nature MI / npj QI / Nature Communications until hardware + theoretical contribution exists.
- **Biggest weakness:** The 2603.10707 collision combined with the simulation‑only execution. Together they reduce the claimable novelty to incremental.
- **Strongest selling point:** The cross‑domain (chaos + econometric) benchmarking with QLIKE/DM/MCS is genuinely uncommon in the QRC literature and is the user's clearest defensible contribution — combined with a Quandela hardware run, this becomes a meaningful paper.

**2‑week action plan (do these in order):**
1. Read and annotate 2603.10707, 2505.13933, Nerenberg 2024–2025, García‑Beni 2023, and Patton & Sheppard 2009 in full.
2. Rewrite the project README and walkthrough.md to remove "50×" and "outperforms" language.
3. Add Random Fourier Features + Ridge baseline of matched feature dimension across all current experiments.
4. Run a full ESN size sweep (50–2000) and recompute matched‑dimension MC.
5. Implement walk‑forward CV for the RV experiment with proper anti‑leakage scaling.
6. Implement Diebold‑Mariano with Newey‑West HAC and Hansen MCS test.
7. Email the Quandela contact and request a concrete hardware quota for Ascella/Belenos with a target experiment size and timeline.
8. Draft the "Concurrent independent work" paragraph for 2603.10707 explicitly.

**2‑month action plan:**
1. Complete the workshop‑grade paper (Tier A) — 8–10 pages, full ablations, DM/MCS, IPC analysis, no hardware needed.
2. Submit to QTML 2026 *and* one workshop (NeurIPS ML4PS or QML).
3. In parallel, run Quandela Ascella experiments at small (n,m) on a subset of RV days; produce sim‑vs‑hardware concordance plot.
4. Reproduce Li et al. 2505.13933's Ising‑QRC on the same RV split, even with a small qubit count, to establish a fair quantum cross‑platform comparison.
5. Recruit at least one econometrics co‑author or advisor — the financial evaluation rigor will land much better with that signature.
6. After hardware results, draft Tier B paper aimed at *Quantum Machine Intelligence* or *Quantum Science and Technology*; expect a 4–6 month review cycle.
7. Prepare a hardware‑forward Tier C version if the Ascella/Belenos run yields a clean, scalable demonstration; aim for npj Quantum Information no earlier than late 2026 / 2027.

## Recommendations

- Treat 2603.10707 as a *concurrent independent* paper, not a competitor to outdo — explicitly cite, distinguish on the temporal/econometric/hardware axes, and let the academic community see a coherent line of work.
- Reframe the central contribution from "new architecture" to "first econometrically rigorous photon‑ensemble QRC benchmark across chaos and volatility," which is *defensible* and *true*.
- Drop or rewrite every order‑of‑magnitude claim ("50×", "outperforms") with matched‑dimension comparisons; if HPT‑QRC ties HAR on QLIKE that is *publishable and honest*.
- Add Random Fourier Features + Ridge as the must‑have classical comparator. If you cannot beat or tie it, the paper's framing must shift to "quantum features as a hardware‑native alternative" rather than "quantum features are better."
- Get on Quandela hardware before a journal submission — the gap between simulation‑only and hardware photonic QRC is the single biggest leverage point for venue tier.
- Submit a workshop paper *now* (QTML 2026 / NeurIPS QML / ML4PS) to plant a flag, then iterate to a journal version once hardware is in. **Submitting a workshop paper does not preclude a fuller journal version later** — this is standard in the QML community.

## Caveats

- I was unable to directly fetch the project website (qedi‑qpfl.vercel.app), the GitHub repository, or `multi_qrc.py`/`walkthrough.md` due to fetch restrictions; the project description above is reconstructed from the user's task brief, the published EPFL/Quandela hackathon‑winner LinkedIn post (which confirms the "HPT‑QRC, MerLin‑powered, swaption volatility surface, beat LSTM" framing), and the very close concurrent preprint 2603.10707 (which appears to share the EPFL/Quandela hackathon ecosystem and shares essentially the entire architecture). If specific code details differ — e.g., HPT_QRC_Multi uses *recurrent* feedback rather than a sliding window — several specific recommendations (e.g., ESP analysis becomes mandatory, kernel‑map framing becomes inappropriate) would change accordingly. Verify these technical details before finalizing the paper.
- The Amanov/Azamov preprint may or may not undergo peer review acceptance; even if it is currently arXiv‑only, by the time HPT‑QRC is submitted it will have been on arXiv for months and reviewers will have seen it. The novelty calculus does not depend on its journal acceptance.
- The "1st place at quantum hackathon" provides credibility *signal* but not *novelty*; reviewers (correctly) discount hackathon awards entirely.
- Q1 vs. workshop is a function of the hardware results that do not yet exist; my recommendation is conditional on (i) actually obtaining Quandela cloud time, (ii) the simulation→hardware concordance being reasonable. If Quandela hardware is unavailable in the next 6 months, the workshop path is unambiguously the right one and the user should not delay for a Q1 attempt.
- I have flagged specific arXiv numbers and authors as confidently as the snippets allow; before submission, every cited paper should be re‑verified against its canonical version. In particular some 2025–2026 dates straddle versions (v1 vs. v2) and the user should always cite the most recent peer‑reviewed version available.