# Objective 3 — Design Specification: Cross-Domain Feature Contract

**Design audit only. No transfer code written and no model trained.** The
Brazil-side source-performance gate is **answered from frozen Objective 1
outputs without retraining**, via the information-equivalence argument in Part
5c — established from the saved pipeline, not from feature names. Objectives 1
and 3 are frozen and untouched.

Generated: 2026-09-06T16:22:41

## Framing

Objective 3 is **cross-domain compatibility and potential-fault screening**:
whether a classifier learned from the labelled Brazil dataset produces
meaningful screening behaviour when applied to Hong Kong, while explicitly
recognising that HK predictions cannot be checked against truth. HK has no
per-timestep fault labels, so no statement in this document asserts that any HK
prediction is correct.

---

## Part 1 — Semantic comparability

Full table with reasoning in `objective3_semantic_comparability.csv`. Summary:

| Quantity | Classification |
|---|---|
| Power (capacity-normalised) | PROXY-COMPARABLE ONLY — DC vs AC, not physically equivalent |
| Irradiance | PROXY-COMPARABLE ONLY — plane and station pairing unresolved |
| Module / ambient temperature | NOT COMPARABLE |
| Expected power | NOT COMPARABLE (absent in HK by design) |
| Residual / ratio | NOT COMPARABLE (absent in HK) |
| String voltage / current + imbalance | NOT COMPARABLE (absent in HK) |
| Sampling support | NOT COMPARABLE (handled diagnostically in Part 2a) |
| Capacity | PROXY-COMPARABLE ONLY (normalisation basis) |
| Derived clearness-like ratio | UNRESOLVED (candidate, not adopted) |

### Physical equivalence versus proxy comparability

**Neither of the two shared quantities is physically equivalent across the two
domains, and this spec does not claim otherwise.** Brazil power is pre-inverter
DC and HK power is post-inverter AC — different physical quantities measured at
different points in the energy chain, and capacity normalisation rescales
magnitude without reconciling the conversion. Brazil irradiance is panel-plane
at 30 deg tilt from a co-located pyranometer; HK irradiance is of unresolved
plane from a weather station whose pairing to SQ1 is undocumented.

The correct classification for both is therefore **proxy-comparable for
exploratory binary screening, not physically equivalent**. A proxy alignment can
support a Normal-versus-anomalous screening signal, where what matters is
whether output is depressed relative to the same site's own conditioned
expectation. It cannot support quantitative cross-domain physical claims, and it
cannot support fault-type assignment.

### The four problems, addressed

**1. AC/DC.** Brazil power is pre-inverter DC; HK is post-inverter AC.
Capacity normalisation puts both on a 0-1 scale but does not reconcile the
physics: it rescales magnitude, not the conversion. Closing the gap would need
either DC-side measurement at SQ1 — recorded nowhere in the HK release, which
carries only `dcVoltage(V)` and no DC current — or a load-dependent inverter
efficiency curve for the SolarEdge unit, which the metadata does not publish.
An assumed fixed efficiency is rejected here for the same reason Objective 2
rejected it. **The residual is left explicitly unresolved and travels with every
result.**

**2. Temperature.** Rear-module temperature runs roughly 15-25 C above ambient
under load, and the offset varies with irradiance and wind rather than being a
subtractable constant. No defensible mapping exists without a thermal model
carrying new unjustified parameters. **Temperature is excluded from the shared
contract.**

**3. String signals — the decisive finding.** Objective 1 established that
removing string voltage collapsed short-circuit F1 to **0.002**, and that
voltage removal cost roughly twice what current removal cost (0.310-0.377
against 0.124-0.181 across four model families). Those findings rest entirely on
`vdc1`, `vdc2`, `idc1`, `idc2` and their derived imbalance features. **Hong Kong
records none of them at any station.**

Stated precisely: the features carrying Brazil's **strongest** fault
discrimination are unavailable in the target domain. This is not the claim that
all Brazil discrimination came from string-level signals — it did not. The
reduced aggregate representations retained non-trivial classification ability:
`Computed power + measured irradiance` reached Macro F1 **0.6938**
(MLP) and `Computed total power only` **0.3635** (HistGBM), against
**0.9852** for the full electrical set — all read directly from
`extended_ablation_summary.csv`. A transferred classifier restricted to the
shared contract therefore does not start from zero, but it does lose the signal
that lifted the source task from moderate to near-ceiling separability, and it
loses it disproportionately on the classes named in Part 3.

Those figures are context for the size of the loss. `Computed power + measured
irradiance` is additionally shown in Part 5c to be **information-equivalent to
the shared contract for the actual saved MLP pipeline**, which is what allows it
to serve as the source-performance gate without retraining. That equivalence is
established from the pipeline, not from the feature names, and it does **not**
extend to `POWER_PHYSICS` or any other nearby set.

**4. Sampling rate.** Brazil is 1 Hz; HK is 15-minute means. A 10-minute
induction is ~600 Brazil samples but sits **inside a single HK interval**, where
it would appear only as a partial depression of one averaged value. A
transferred classifier could therefore not resolve a Brazil-style short
induction in HK at all. Only conditions persisting across multiple 15-minute
intervals — degradation, extended shading, sustained damage — are resolvable.

---

### 2b. Joint two-feature support

Marginal statistics are not sufficient, because the classifier consumes the two
features **jointly**. The method below was fixed before any result was seen: two
descriptive containment measures, no learned domain classifier, no KDE
classifier, no fitted predictive model of any kind. HK is the clean-sensor /
no-known-fault 2023 population (n = 4,745 rows with both quantities
present).

| Brazil representation | Capacity | HK inside Brazil convex hull | HK inside Brazil-occupied cells (20x20) |
|---|---|---|---|
| All classes, native 1 Hz | 5.00 kW | 95.68% | 89.97% |
| All classes, native 1 Hz | 5.28 kW | 95.49% | 88.24% |
| Normal only, native 1 Hz | 5.00 kW | 94.84% | 69.04% |
| Normal only, native 1 Hz | 5.28 kW | 94.77% | 67.38% |
| All classes, 900-s diagnostic | 5.00 kW | 64.40% | 69.65% |
| All classes, 900-s diagnostic | 5.28 kW | 60.82% | 67.95% |

Full table in `objective3_joint_support.csv`; figure in `objective3_joint_support.png`.

**Which HK population each statistic uses — the two power KS values are not in
conflict.** They are computed on different populations, both legitimately:

| Statistic | HK population | n | Value |
|---|---|---|---|
| Marginal KS, normalised power | HK eligible, **full 2023** | 15,870 | **0.320** |
| Marginal KS, irradiance | HK **clean-sensor months** (gate) | 4,749 | **0.340** |
| Joint-table KS, normalised power | HK **clean-sensor months** (gate) | 4,745 | **0.383** |
| Joint-table KS, irradiance | HK **clean-sensor months** (gate) | 4,745 | **0.340** |

Power does not depend on the corrupted HK irradiance sensor, so its marginal
statistic may legitimately use the larger full-2023 population; irradiance may
not. The joint diagnostic requires **both** quantities present, so it is
confined to the clean-sensor population throughout, and the small 4,749 -> 4,745
difference is the four clean-sensor rows lacking `power_W`. **Restricted to the
same clean-sensor population, the power KS is 0.383, not 0.320.** Every table in
this document names its population; no statistic is quoted without one.

**Both are reported deliberately, and neither alone defines a pass or fail.**
Convex-hull containment is permissive — a hull bridges empty interior regions,
so a point can lie "inside" a region the source never occupied. Occupied-bin
containment is more sensitive to actual joint source occupancy. Gate 2 is not
reduced to either statistic.

**What the two measures show together.** For Brazil all-classes native, the two
agree closely (95.7% hull
against 90.0% bins), so HK
sits inside genuinely occupied source regions rather than inside hull-bridged
voids. The Normal-only comparison is where they separate sharply
(94.8% hull against
69.0% bins): the hull looks
almost as permissive as before, but roughly a third of HK points fall in cells
Brazil's Normal class never occupied. This is exactly the gap the hull measure
conceals, and it is the more relevant number for a screening model whose
reference is Brazil Normal.

**The cross-domain normalised-power-per-irradiance offset.** In the joint plane,
Brazil's Normal class occupies a tight upper ridge while HK sits systematically
below it — less capacity-normalised power at the same irradiance. Restricting to
irradiance above 200 W/m², the median of capacity-normalised power per kW/m² is
**0.967 for Brazil Normal (IQR 0.925-0.990) against 0.799 for HK clean-sensor
(IQR 0.668-0.947)**, a ratio of **0.83** — HK approximately **17% lower** in
this comparison. The convex hull spans the empty region between the two ridges,
which is why hull containment stays high while occupied-bin containment falls.

**The cause of this offset cannot be isolated from the available evidence.** It
may reflect any combination of the AC-versus-DC measurement point, array
mounting and the 30 deg versus 0 deg tilt difference, the unresolved HK
irradiance plane and the undocumented weather-station-to-SQ1 pairing, module and
ambient temperature regimes, soiling, system losses, and other domain
differences not enumerated here. **No inverter conversion efficiency is assumed,
quantified, or attributed** — Objective 2 deliberately declined to introduce one,
and nothing in this audit establishes a value. The offset is **not evidence of a
fault at SQ1.**

**This is the most consequential Gate 2 finding for a screening transfer, and it
is a result to be observed rather than adjusted.** A classifier whose Normal
reference is Brazil's ridge will score a substantial share of ordinary HK
operation as unlike Brazil Normal. That behaviour **is part of the
transfer-compatibility finding**: it is what direct transfer across these two
domains actually does, and suppressing it before measuring it would destroy the
result. See Part 5d for the binding no-target-adaptation rule.

**HK Normal-status caveat.** The clean-sensor HK subset contains no known fault
event, but it carries **no verified per-timestamp Normal labels**. It is
described throughout as "clean-sensor / no-known-fault". All findings here are
stated **relative to Brazil Normal**, and none is a validated comparison between
two confirmed-normal populations.

The 900-s diagnostic rows are reported for completeness. They show
**substantially lower containment**, but that representation simultaneously
changes the temporal support **and** reduces the source sample to 672
blocks. Those two effects cannot be separated from this diagnostic alone, and no
further experiment is run here to separate them. The 900-s rows are therefore
**not used to determine Gate 2**, and no causal reading of the lower containment
is offered in either direction.

---

## Part 3 — What can a Brazil classifier's output mean on HK?

| Class | Physically applicable to SQ1? | Can a prediction be interpreted? |
|---|---|---|
| **Short-Circuit** | The condition can occur | **No.** Assignment rests on string-level voltage behaviour HK does not measure. Any short-circuit prediction would be an extrapolation from aggregate power alone, and Objective 1 showed this class is exactly the one that collapses without voltage. |
| **Open-Circuit** | The condition can occur | **No**, for the same reason. |
| **Degradation** | Yes in principle | **Weakly.** Brazil's was induced by a resistive load bank over ~10 minutes; HK degradation would develop over months. Same label, different physical process and timescale. |
| **Shadowing** | Yes — urban campus with surrounding buildings | **Most defensibly of the four.** Reduced output under partial obstruction is visible in aggregate power, and does not require string measurement. |
| **Normal** | Yes | **Yes.** |

### Recommendation

**Use the output as a binary Normal-versus-anomalous screening signal, and
report the predicted class distribution separately as a diagnostic of transfer
behaviour rather than as a claim about HK conditions.**

Reasoning: two of the five classes cannot be assigned on the available sensing,
and a third maps onto a different physical process. Retaining the fault-type
distinction would imply a discrimination the input features cannot support.
Collapsing to binary keeps exactly the claim the data can carry — that a
timestamp looks unlike Brazil's Normal class — while the class distribution is
still worth reporting because a collapse toward one class is itself the
diagnostic that distinguishes a domain mismatch from a working transfer.

Temporal clustering of flags is retained as a secondary check, since a screening
signal that fires in coherent runs aligned with known HK context is more
credible than one firing at dispersed isolated timestamps.

---

## Part 4 — Reportable metrics without labels

**Computable and reportable:**

1. Predicted class distribution on HK, compared against Brazil's class priors.
2. Temporal clustering of predicted non-Normal flags — dispersed or
   concentrated, and the run-length distribution.
3. Alignment of flags with known HK context: the Typhoon Saola proxy event
   window (1-15 Sept 2023), the March-October 2023 sensor-fault window, and the
   2023 residual shift identified in Objective 2.

   **September lies inside the March-October measured-irradiance corruption
   period.** Because the shared contract includes measured HK irradiance, any
   Objective 3 result over the Saola window is computed on a knowingly corrupted
   input. It can therefore be reported **only as a corrupted-input stress test,
   never as evidence of PV-event detection.** The same caution applies to any
   comparison against Objective 2 screening signals over that window: both
   signals share the same corrupted input, so agreement between them is not
   independent corroboration.
4. Agreement between the transferred classifier's flags and Objective 2's
   anomaly-screening flags — one screening signal against another, **neither
   validated against truth**.

**Not computable, and must not be reported:** accuracy, precision, recall, F1,
or confusion matrices on HK. HK has no per-timestep fault labels, so every one
of these would presuppose a ground truth that does not exist.

A banned-phrase guard is specified for the implementation stage, covering the
8 label-presuming phrases enumerated in the `BANNED` constant in
`objective3_feature_audit.py`. They are deliberately not reproduced here: this
document is itself passed through the guard, and listing them would trip it on
its own prose. The guard is already active in this audit script.

---

## Part 5 — Proposed minimal experiment

### 5a. Shared feature set

> ### ⚠ Headline finding — the shared contract is 2 features
>
> **Capacity-normalised power, and irradiance. Nothing else survives the
> intersection.** This is below the prompt's three-feature threshold, and
> neither of the two survivors is clean:
>
> 1. **Power carries an unresolved DC-versus-AC mismatch.** Brazil's
>    `p_dc_computed` is DC string power; HK's `power_W` is AC active power at
>    the inverter output. No inverter efficiency was assumed to bridge them, so
>    the gap is carried, not closed.
> 2. **HK's irradiance plane is unresolved** — tilted-plane versus horizontal is
>    undocumented, as is the weather-station-to-SQ1 pairing.
> 3. **The discrimination that mattered most is absent.** Objective 1 showed the
>    strongest fault discrimination — Short-Circuit above all — depended heavily
>    on string-level voltage and current that HK does not record at any station.
>
> Taken together, **this makes the semantic gate marginal before distribution
> overlap is even considered.** Two features, one semantically mismatched, one
> geometrically undocumented, and the classes of interest stripped of the
> measurements that separated them.

Whether the experiment remains worthwhile on those terms is addressed in the
gate summary below and in `objective3_open_decisions.md`; it is not a decision this
audit takes alone.

The derived clearness-like ratio (normalised power / irradiance) is available as
a third candidate but is not adopted here, since it inherits both parents'
caveats and adds no independent information.

### 5b. Normalisation

Power divided by rated capacity — Brazil by 5.00 kW with 5.28 kW carried as a
sensitivity, HK by 27.6 kW from the SQ1 TTL entry. Irradiance is left in native
W/m², since both are already the same unit. Any further scaling (standardisation)
is fitted on **Brazil training data only** and applied unchanged to HK.

#### Target-side handling of HK power — required, and easy to get wrong

The frozen `POWER_PLUS_IRR` configuration fits its `SimpleImputer` and
`StandardScaler` on **raw Brazil watts** (`p_dc_computed`), not on a normalised
quantity. **Feeding HK power divided by 27.6 kW into that parameterisation would
be wrong** — the scaler expects values on a Brazil-watt scale and would receive
values near 0-1, standardising them to a large constant negative offset and
destroying the transfer.

The correct target-side step expresses HK power in **Brazil-capacity-equivalent
watts** before the Brazil-fitted preprocessing:

```
p_hk_brazil_equivalent_W = power_W * (CAPACITY_BR / CAPACITY_HK)
                         = power_W * (5000.0 / 27606.0)     # = 0.18112005
                         = power_W * (5280.0 / 27606.0)     # = 0.19126277  (5.28 kW variant)
```

Irradiance (`irr_meas`, W/m²) is passed through **unchanged**, since Brazil's
`irr_panel_wm2` is already in native W/m².

This is exactly equivalent to the capacity-normalised formulation, because the
Brazil-fitted scaler is affine. Writing `mu`, `sd` for the Brazil training-fold
mean and standard deviation of raw watts:

```
(p_hk/C_HK - mu/C_BR) / (sd/C_BR)  ==  (p_hk*(C_BR/C_HK) - mu) / sd
```

The left side is the shared-contract view (both domains capacity-normalised, the
scaler refitted on Brazil normalised power); the right side is what the frozen
raw-watt parameterisation computes when fed Brazil-capacity-equivalent watts.
They are algebraically identical, and were verified numerically to agree to
within 8.9e-16. **Use the right-hand form**, because it preserves the frozen
parameterisation exactly and requires nothing to be refitted.

Note the direction of the conversion: HK power is scaled **into** Brazil's
capacity frame, not the reverse. This carries no assumption about conversion
efficiency or system losses — it is a unit change on the capacity axis only.

### 5c. Model and the source-performance gate

**Source-performance gate: EVIDENCED from frozen outputs, without retraining.**

The earlier draft of this spec held the gate PENDING on the ground that no
nearby configuration should be substituted for the untested shared contract.
That reasoning was right in general but wrong in this specific case, and the
difference was established by reading the pipeline rather than the feature name.

Objective 1's `Computed power + measured irradiance` is defined in
`extended_ablation.py` as `POWER_PLUS_IRR = ["p_dc_computed", "irr_panel_wm2"]`.
The proposed shared representation is `p_dc_computed / capacity` and
`irr_panel_wm2`. The two differ only by a **constant positive divisor on one
feature**, and the saved winning model is MLP, which is in `SCALE_SENSITIVE`,
so a fold-fitted `StandardScaler` sits between the raw feature and the model:

| Pipeline stage | Effect of dividing one feature by a constant C > 0 |
|---|---|
| `inf -> NaN` | unchanged — `x/C` is finite exactly when `x` is |
| `SimpleImputer(median)`, training fold only | `median(x/C) = median(x)/C` — equivariant, same point in scaled space |
| `StandardScaler`, training fold only | `(x/C - mean(x)/C) / (sd(x)/C) = (x - mean(x)) / sd(x)` — **C cancels exactly** |
| Fold allocation, seed, hyperparameters, MLP downsampling indices | identical; none depends on feature scale |

The matrix reaching the MLP is therefore **mathematically identical**, so the
two representations are information-equivalent for the actual saved pipeline.
The equivalence is a property of this pipeline, not of the feature list: for a
model **not** in `SCALE_SENSITIVE` the argument would rest instead on tree
invariance to monotone transforms, and for a pipeline without a scaler it would
not hold at all. The one honest caveat is floating point — `x/C` then
standardise is exact in real arithmetic but not guaranteed bit-identical in
IEEE-754, with relative differences at machine epsilon.

Accordingly the **five-class** component of the gate is answered by the frozen
result **Macro F1 0.6840 (MLP)**, obtained under the identical group-disjoint
5-fold protocol and fold allocation. The exclusion of `POWER_PHYSICS` and other
nearby sets still stands — those genuinely add physics-derived or environmental
inputs the contract excludes.

#### No fitted model artefact survives — what implementation must therefore do

A read-only search of the project for serialised model artefacts (`.pkl`,
`.joblib`, `.pt`, `.h5`, `.keras`, `.sav`, `.onnx`) returns **three files, all
irrelevant**: `lstm_model.pt`, `lstm_no_sensor.pt` and `lstm_with_sensor.pt`,
dated 2026-07-03, all under `outputs/lstm_with_sensor/` and `outputs/no_sensor/`
— Hong Kong LSTMs from the historical pipeline that Objective 2 established was
contaminated, and which are on this audit's forbidden-input list. They are not
Brazil models, not `POWER_PLUS_IRR`, and not usable.

`extended_ablation.py` and `pv_fault_classifier.py` contain **no model
persistence at all** — no serialisation call of any kind: no `joblib`, no
`pickle`, and no deep-learning checkpoint write. What
survives from Objective 1 is out-of-fold predictions, per-fold and per-class
metrics, and confusion-matrix counts.

Therefore **no fitted, transferable `POWER_PLUS_IRR` MLP exists and none can be
recovered.** Objective 3 implementation must fit the already-fixed Brazil
`POWER_PLUS_IRR` configuration — same feature list, same fold allocation, same
seed, same hyperparameters, same imputer and scaler discipline. This is
**implementation of a frozen design, not new model selection.** No architecture,
feature set, hyperparameter or protocol choice is reopened, and the fitted result
must reproduce the frozen five-class Macro F1 of 0.6840 as a correctness check
before any HK data is touched.

#### The primary HK endpoint is locked

The transferred model is the **same five-class source model**, with its
predictions **collapsed to Normal versus non-Normal for interpretation**. The
collapse happens at the prediction stage, downstream of the model.

**A newly designed binary classifier must not be substituted.** The Gate 3
evidence rests on an information-equivalence argument about one specific frozen
five-class configuration and its saved out-of-fold predictions. Training a
binary model instead would be a different configuration whose performance is not
evidenced by anything frozen, and it would break the equivalence that
established Gate 3 in the first place. The collapsed binary figures in this
document describe the existing five-class model interpreted as Normal versus
non-Normal — they are not, and must not be reported as, results from a
separately trained binary classifier.

### 5d. HK application

Objective 2's sensor-free eligibility mask (`ghi_clear >= 50 W/m²` and
`solar_zenith < 85°`).

#### Binding rule: the primary experiment is TRUE DIRECT TRANSFER

**No target-domain adaptation of any kind.** The primary transferred classifier
uses **Brazil-fitted preprocessing unchanged on HK**: the Brazil training-fold
median imputer, the Brazil training-fold `StandardScaler`, and the Brazil-fitted
model, with HK power converted to Brazil-capacity-equivalent watts by the fixed
constant in Part 5b and irradiance passed through in native W/m². Nothing is
refitted, recentred, rescaled or recalibrated on HK.

This is binding and rules out, for the primary result: recentring the
normalisation on HK's own operating distribution; standardising HK against HK
statistics; threshold calibration against an HK baseline; and any quantile,
mean or variance matching between the domains. All of these are target-domain
adaptation and are **out of scope for Objective 3.**

The cross-domain normalised-power-per-irradiance offset documented in Part 2b is
**part of the transfer-compatibility result, not a prerequisite to resolve before
observing transfer behaviour.** Correcting for it in advance would mean reporting
the behaviour of an adapted model while describing it as direct transfer.

An HK-relative descriptive comparator — for example flag rates referenced to
HK's own operating distribution — **may be reported separately** and is useful
context. It must not alter the features or preprocessing supplied to the
transferred classifier, and it must never be presented as direct transfer.

**Because the shared contract includes measured HK irradiance, the primary
transfer application is restricted to the clean-sensor periods — January,
February, November and December 2023.** Full-2023 application may be reported
only as a separately labelled corrupted-input stress test, never as the primary
result. Feeding a knowingly corrupted sensor signal into the primary experiment
would confound target-sensor failure with transfer failure.

### 5e. Sampling reconciliation

Three options:

| Option | Effect |
|---|---|
| Train on native 1 Hz Brazil, apply to HK 15-min | Preserves all Brazil fault events and the full label set, but source and target see different temporal support; the model meets narrower-variance inputs at application time |
| Aggregate Brazil to 900-sample blocks and train on those | Matches support, but reduces Brazil from 613,711 rows to **456 usable labelled blocks** (see the continuity accounting below), and a ~600-second induction can sit entirely inside one 900-second block |
| Train on native, apply to HK, and report the Part 2a diagnostic alongside | Keeps the source task intact while quantifying how much of the observed distribution discrepancy is support-driven |

#### Continuity accounting for the 900-second blocks

`fault_dataset.csv` is a **filtered** dataset, so `sample_index`
contains gaps and 900 consecutive dataframe rows do not necessarily span 900
seconds. Blocks are therefore built by splitting first and aggregating second:
each recording group is divided into maximal contiguous `sample_index` runs
(cut wherever `diff() != 1`), and non-overlapping 900-sample blocks are formed
**inside** each run, discarding only the trailing remainder of each run. Every
retained block is asserted to belong to one recording group, lie within one
contiguous run, contain exactly 900 rows, and satisfy `np.diff(sample_index)==1`
throughout. Blocks lost to filtered-data gaps are **not** counted as valid
temporal-support matches.

| Quantity | Count |
|---|---|
| Recording groups | 16 |
| Contiguous `sample_index` runs | 114 |
| Discontinuities inside groups | 98 |
| 900-row blocks under naive row-based chunking | 675 |
| **Valid continuous 900-second blocks retained** | **672** |
| — of which mixed-label (left unlabelled) | 216 |
| — of which single-label, usable for supervised aggregation | **456** |

**Implication for temporal aggregation as a transfer-training strategy.**
Matching HK's 900-second support costs the source task 3
blocks to continuity alone and a further 216 to label mixing, leaving
456 labelled training rows drawn from 16 recording groups —
too few to support the group-disjoint 5-fold protocol at anything like the
frozen Objective 1 statistical resolution, and far too few for five classes.

The class composition of the survivors is decisive, and it was verified
directly from the post-continuity blocks rather than inferred:

| Class | Pure 900-s blocks | Longest event in register |
|---|---|---|
| Normal | 323 | 56,719 samples |
| Shadowing | 130 | 6,555 samples |
| Degradation | **3** | 2,695 samples |
| Short-Circuit | **0** | **619 samples** |
| Open-Circuit | **0** | **606 samples** |

The mechanism is simply that **standard controlled inductions are ~600 seconds,
which is shorter than a 900-second support block.** A short event cannot
generate a pure block at all: any block overlapping it necessarily also contains
Normal seconds, so it carries two distinct `f_nv` values, is mixed-label, and is
excluded by the existing rule. There is no separate dilution effect to invoke —
a 600-second fault sitting inside a 900-second block *is* a mixed-label block by
construction. Temporal aggregation therefore removes short controlled events
from the usable supervised set entirely.

The only controlled-class survivors are the **3 Degradation blocks**, and they
trace to the two anomalous long-duration degradation events in the register
(event 1272, 1,779 samples; event 1430, 2,695 samples) — the only controlled
events anywhere in the dataset exceeding 900 samples. They fall in just 2 of the
16 recording groups.

**Grouped five-class training on 900-second blocks is therefore not merely
under-powered — it is impossible.** Two of the five classes have no training
examples at any capacity, and a third has three examples spanning two groups,
which cannot survive a group-disjoint 5-fold split.

**Recommendation: the third option.** Train on native 1 Hz Brazil, apply to HK,
and carry the Part 2a block statistics as a reported diagnostic of how much of
the observed distribution discrepancy is support-driven rather than domain-
driven. Note also that Brazil has no timestamps, so any aggregation is by sample
position within a recording group, never by clock.

### 5f. The three gates

| Gate | Source | Status |
|---|---|---|
| **Semantic** | Part 1 | **FAILS for fault-type transfer, PARTIAL for binary screening.** Neither shared quantity is physically equivalent — both are **proxy-comparable only** (DC vs AC; unresolved irradiance plane and station pairing). Two of five classes cannot be assigned on the available sensing, and the string features carrying the strongest discrimination are absent. |
| **Domain overlap** | Part 2 | **MARGINAL — a qualitative design judgement, not a predefined statistical threshold.** No pass/fail cut-off is attached to any KS or overlap value, and Gate 2 is not reduced to a single statistic. Populations differ by quantity and are named individually in Part 2 and 2b: power on HK eligible full 2023, irradiance on HK clean-sensor months. Full-2023 measured irradiance is excluded as a sensor-corruption diagnostic. See `objective3_distribution_comparison.csv`, `objective3_joint_support.csv` and the figures. |
| **Source performance** | Part 5c | **EVIDENCED, no retraining required.** Five-class Macro F1 **0.6840**; collapsed binary anomalous F1 **0.8268**, balanced accuracy **0.8749**. Both read from frozen Objective 1 outputs under the identical grouped protocol, via a verified information-equivalence argument — not inferred from feature names. |

### 5g. Distinguishing a negative result from a broken mapping

Specified **before anything is run**, because it is the crux of whether a
negative result is reportable:

1. **Source-performance floor.** If the two-feature Brazil model cannot itself
   discriminate on Brazil under the frozen protocol, a failed HK transfer says
   nothing about the domains — the mapping never carried discrimination to
   begin with. This is why gate 3 must be settled first and must not be
   estimated.
2. **Sanity transfer within Brazil.** Apply the restricted model across
   held-out Brazil recording groups. Success there with failure on HK isolates
   the domain change; failure in both indicates the feature set, not the domain.
3. **Input-range audit at application time.** Record the fraction of HK rows
   falling outside Brazil's observed support per feature. Widespread
   extrapolation means the classifier is being asked about inputs it never saw,
   which is a mapping problem rather than evidence about HK.
4. **Degenerate-output check.** If the predicted distribution collapses to a
   single class, distinguish collapse-to-Normal (consistent with HK genuinely
   operating normally) from collapse-to-a-fault-class (a strong indicator of
   distribution shift rather than a finding about HK).
5. **Corrupted-input contrast.** Compare the clean-sensor primary result with
   the full-2023 stress test. A large divergence attributable to the known
   March-October corruption is a target-sensor artefact, not a transfer result.

**Reading the full-2023 irradiance KS correctly.** The full-2023 comparison
gives a *lower* KS (0.276) than the clean-sensor gate comparison (0.340). This
must not be read as better physical domain agreement, and equally must not be
asserted to be definitely caused by corruption. The defensible statement is
narrower: the full-2023 population **includes the known March-October
corrupted-sensor period, so its KS cannot be interpreted as evidence of better
physical agreement between the domains.** The comparison is not clean enough to
support an interpretation in either direction, which is precisely why it is
excluded from the gate.

Only if the source-performance floor is cleared, the within-Brazil sanity
transfer succeeds, and extrapolation is limited can a negative HK outcome be
attributed to genuine domain difference.

---

## Classification of findings

CONFIRMED: the AC/DC asymmetry, the absence of HK string measurement, the
sampling-support mismatch, the capacity figures, the March-October sensor
corruption, and Objective 1's string-signal results.
SUPPORTED ASSUMPTION: irradiance comparability (HK plane inert at tilt 0), the
weather-station pairing to SQ1.
UNRESOLVED: the DC/AC residual magnitude, the derived clearness-like ratio's
usefulness, and the source-performance gate.
