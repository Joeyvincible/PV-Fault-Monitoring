# Objective 3 — Open Decisions

Requiring approval before any implementation. This audit trained nothing and
takes none of these decisions unilaterally.

## 1. Does a two-feature contract justify proceeding at all?

The shared contract is **capacity-normalised power and irradiance**. Everything
carrying Brazil's fault discrimination — string voltage, string current, their
imbalance ratios, expected power, residual and ratio — is absent in Hong Kong.

Three courses:

| Option | What it yields |
|---|---|
| Do not proceed | Objective 3 is reported as a design-stage negative result: the cross-domain feature contract is too thin to support fault-type transfer, evidenced rather than asserted |
| Proceed as a documented negative result | Run the minimal experiment expecting failure, with Part 5g in place so the failure is attributable |
| Proceed as binary screening only | Discard fault-type prediction, transfer a Normal-versus-anomalous signal, report agreement with Objective 2 screening |

**Recommendation: the third — binary screening — and it is now actionable.**
The source-performance gate is settled from frozen evidence (decision 2), and it
lands more favourably for binary screening than the five-class figure suggested:
collapsing the same model to Normal-versus-anomalous gives anomalous F1 0.8268
and balanced accuracy 0.8749, against a five-class Macro F1 of 0.6840. The
five-class weakness is concentrated in *distinguishing fault types* — precisely
the capability Part 3 already rules out on HK — rather than in separating normal
from anomalous, which is the capability the transfer endpoint actually needs.

Option one (do not proceed) is no longer the strongest reading, because the
mapping demonstrably carries screening discrimination on the source domain. A
failed HK transfer would therefore be attributable rather than uninterpretable,
which is what makes a negative result publishable.

## 2. Source-performance gate — RESOLVED, no longer an open decision

This no longer requires training anything. `Computed power + measured
irradiance` is information-equivalent to the shared contract for the saved MLP
pipeline (a constant positive divisor that the fold-fitted StandardScaler
cancels exactly), so the frozen Objective 1 outputs already answer it:

- **five-class** Macro F1 **0.6840**
- **collapsed binary** (Normal vs anomalous) anomalous F1 **0.8268**, balanced
  accuracy **0.8749**, Normal recall **0.8703**, anomalous recall **0.8794**,
  anomalous precision **0.7801**

Both come from the pooled out-of-fold predictions under the identical
group-disjoint protocol. The binary figures describe the **existing five-class
model interpreted as Normal vs non-Normal** — not a separately trained binary
classifier. No numerical pass mark is invented here: the evidence is reported
and final acceptance remains an implementation decision.

## 2b. Which Brazil capacity is used is now the live normalisation question

See decision 4 — it moves from a formality to a real choice only for the joint
support diagnostic, since the source-performance gate is invariant to it.

## 3. Is the derived clearness-like ratio admitted?

Adding `normalised power / irradiance` would make the contract three features.
It adds no independent information and inherits the DC/AC residual, but may help
a low-capacity model. Not adopted here.

## 4. Which Brazil capacity is authoritative?

5.00 kW (paper) or 5.28 kW (16 x 330 W nameplate). Objective 1 found results
insensitive (Macro F1 difference 0.0069), and the source-performance gate is
mathematically invariant to the choice, since the divisor cancels in the
scaler. The choice does move the joint-support diagnostic, but only slightly —
hull containment 95.68% vs 95.49% and occupied-bin containment 89.97% vs 88.24%
for Brazil all-classes. **It does not change Gate 2.** Both are carried in the
comparison files; one must still be fixed for implementation.

## 5. Does binary screening still need the fault-type head?

If the recommendation in Part 3 is accepted, the Brazil model could be trained
directly as a binary Normal-versus-anomalous classifier rather than five-class
and collapsed afterwards. Training binary directly is cleaner; collapsing
afterwards preserves the class distribution as a transfer diagnostic. This audit
assumes five-class-then-collapse, to keep the diagnostic.
