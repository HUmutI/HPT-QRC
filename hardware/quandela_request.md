# Quandela Cloud Quota Request — Draft

**Status:** Draft for Team Qedi to review and send.
**Recipients (suggested):** `support@quandela.com`, plus the EPFL Quantum Hackathon 2026 Quandela challenge contacts.
**Goal:** Obtain Ascella (and/or Belenos) cloud time sufficient for a sim-vs-hardware concordance figure plus a small end-to-end walk-forward window for HPT-QRC, in support of a Q1 journal submission.

---

## Subject

EPFL Hackathon 2026 winner — Quandela cloud-time request for HPT-QRC photonic-reservoir paper (≈ 200 inferences at (n,m)=(3,6) and (4,8))

---

## Body (draft)

Dear Quandela team,

We are Team Qedi from the EPFL Quantum Hackathon 2026, where our Hybrid Photonic Temporal Quantum Reservoir Computing (HPT-QRC) model placed first in the Quandela Challenge. We are now extending the work into an academic paper, with target venues in the Q1 range (*Quantum Machine Intelligence*, *Quantum Science and Technology*, or *Physical Review Applied*) once a hardware execution is in.

The model is a fixed-encoding linear-optical reservoir built on Perceval + MerLin, with a heterogeneous photon-number ensemble (`photon_list = [2, 3, 4]`) and a Ridge readout. We currently simulate the Fock-state probabilities via Perceval's SLOS backend and benchmark on chaotic (NARMA-10, Mackey-Glass) and volatility (S&P 500 RV, VIX) time-series tasks with full econometric evaluation (Diebold–Mariano with Newey–West HAC, Hansen MCS, QLIKE, IPC).

We would like to request cloud quota on **Ascella** (and optionally **Belenos**) for:

1. **Concordance experiment.** ≈ 200 inputs per configuration at `(n_photons, n_modes) ∈ {(3, 6), (4, 8)}`, ≈ 10² – 10³ shots per input. Used to produce a simulation-vs-hardware Fock-probability scatter and per-mode TVD figure.
2. **End-to-end walk-forward window.** Last ≈ 60 trading days of 2024 on S&P 500 RV, single photon configuration `(n, m) = (3, 6)`, shot count to be tuned based on (1). Used to demonstrate sub-second-latency inference on real hardware.

We are flexible on:
- Photon and mode count (we can downscale to whatever the platform supports cleanly).
- Shot count and runtime trade-offs (we will run a shot-count sensitivity sweep on simulation first so we land on hardware with informed defaults).
- Time window (anything in the next 8–10 weeks works for our journal timeline).

We are happy to acknowledge Quandela explicitly in the paper, share the hardware results back with you for any case studies, and to coordinate with the Perceval / MerLin teams on any platform-specific calibration that helps both sides.

A concurrent independent preprint (Amanov & Azamov, arXiv:2603.10707) has appeared on a related architecture for swaption surfaces; our work distinguishes itself on the temporal axis, on cross-domain benchmarking, on econometric evaluation rigor, and — most importantly — on real hardware execution, which is the missing piece in the current literature.

Please let us know what process is right on your side (form / quota request portal / direct cloud credentials), and what photon/mode envelope and shot budget you can offer.

Thank you very much for your support, and for the great challenge at EPFL.

Best regards,

**Team Qedi**
- Eren Aslan
- Hüseyin Umut Işık
- Arda Kara
- Mehmet Alp Özaydın

EPFL Quantum Hackathon 2026 · Quandela Challenge — 1st place
Repo: <to-fill>

---

## Internal checklist before sending

- [ ] Replace `<to-fill>` repo link with public GitHub URL.
- [ ] Confirm hackathon Quandela contact name and CC them.
- [ ] Attach a 1-page PDF abstract of the workshop paper (once `paper/workshop_draft/` is in shape).
- [ ] Mention any prior Quandela cloud usage from the hackathon period for context.
- [ ] Have at least one team member set up Perceval `RemoteProcessor` credentials in advance so we can act fast once quota is granted.

## Correspondence log

| Date | Direction | Summary |
|---|---|---|
| — | — | — |
