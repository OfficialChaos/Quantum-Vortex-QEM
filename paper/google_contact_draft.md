# Google Quantum AI — Contact / Submission Draft
# Send via: https://quantumai.google/contact
# Or target: mitiq@unitary.fund (Mitiq maintainers — better first contact)

---

## OPTION A: Mitiq Team (Recommended First Contact)
**To:** mitiq@unitary.fund
**Subject:** Contribution proposal: Fibonacci ZNE schedule + stability filter

---

Hi Mitiq team,

I'm an independent researcher and wanted to share a small contribution
that may be of interest to the Mitiq project.

I've implemented and benchmarked two additions to the ZNE pipeline:

1. **Fibonacci noise scaling schedule** — scale factors {1, 2, 3, 5, 8, ...}
   as an alternative to linear/odd schedules. Motivated by the golden-ratio
   spacing properties used in dynamical decoupling pulse sequences.

2. **Fluid-dynamic stability filter** — a post-hoc reliability criterion
   based on the Navier-Stokes energy dissipation norm applied to ZNE
   extrapolation residuals, to flag potentially hallucinated corrections
   without additional circuit evaluations.

Both are implemented against the Cirq/Mitiq stack and evaluated across
circuit depths and noise levels with shot-based simulation. A preprint
is in preparation for arXiv (quant-ph).

Code: https://github.com/OfficialChaos/Quantum-Vortex-QEM

Happy to share the preprint draft or discuss whether this fits the
project's roadmap as a contributed module or example notebook.

Best,
Shawn G. Kleipe
ORCID: 0009-0002-2480-2430

---

## OPTION B: Google Quantum AI (After arXiv Preprint is Live)
**To:** Via https://quantumai.google/contact
**Subject:** Independent research: stability filtering for ZNE — arXiv:XXXX.XXXXX

---

Hello,

I'm writing to share a recent preprint that may be relevant to the
Cirq/Mitiq ecosystem and ongoing QEM research at Google Quantum AI.

**Title:** Fluid-Dynamic Stability Filtering for Zero-Noise Extrapolation
in NISQ-Era Quantum Circuits

**arXiv:** [link once live]
**GitHub:** https://github.com/OfficialChaos/Quantum-Vortex-QEM

The work introduces a Fibonacci noise scaling schedule and a
Navier-Stokes-inspired stability filter for ZNE, both implemented
and benchmarked using Cirq. The stability filter adds O(K) overhead
(K = number of scale factors) and requires no additional circuit
evaluations — it operates on the same expectation values already
collected during ZNE.

I'd welcome any feedback, and would be glad to discuss whether
this could be useful to the team's work.

Best,
Shawn G. Kleipe
ORCID: 0009-0002-2480-2430

---

## SEND ORDER (important)
1. First: get arXiv preprint live (gives you credibility and a citable link)
2. Second: contact Mitiq team (they are responsive and community-oriented)
3. Third: contact Google Quantum AI directly (after Mitiq acknowledgment)

Do NOT send before the arXiv preprint is live — the link is your
credibility anchor. Without it, this reads as unsolicited ideas.
With it, it reads as a researcher sharing work.
