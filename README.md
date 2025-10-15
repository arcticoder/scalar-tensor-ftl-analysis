# Scalar-Tensor FTL Analysis (Phase B)

**Status**: ⏳ ACTIVE RESEARCH (Oct 2025)  
**Question**: Can scalar-tensor gravity theories enable FTL without ANEC/QI violations?  
**Timeline**: 4-6 weeks (mid-November 2025 target)

## Motivation

**Phase A Result** (lqg-anec-framework): FTL is impossible in pure GR+QFT
- All warp metrics violate ANEC and/or Quantum Inequalities
- Gap is insurmountable: 10²³× QI violation margin

**Phase B Question**: Can scalar-tensor theories provide "screening"?

Scalar-tensor theories modify GR through scalar field φ(x):
- Brans-Dicke: G_eff → G/φ (variable gravitational constant)
- Horndeski: Most general scalar-tensor with second-order equations
- DHOST: Degenerate higher-order scalar-tensor theories

**Hypothesis**: Scalar field might screen negative energy or modify ANEC

## Research Plan

### Week 1-2: Brans-Dicke Theory

**Goal**: Test whether variable G_eff = G/φ helps FTL

**Framework**:
```
Action: S = ∫ [φR - ω/φ (∇φ)² + L_matter] √-g d⁴x

Modified Einstein equations:
G_μν = (8πG/φ) T_μν + T_μν^(scalar)

Scalar field equation:
□φ = (8πG/(3+2ω)) T
```

**Tests**:
1. Alcubierre metric with dynamic φ(r_s, t)
2. ANEC computation with scalar contribution
3. QI bounds with φ-field fluctuations

**Expected Result**: Likely still violates (scalar adds T_μν^(scalar) but doesn't change sign)

### Week 3-4: Horndeski Theory

**Goal**: Test most general scalar-tensor with screening

**Framework**:
```
L = ∑_{i=2}^{5} L_i(φ, ∂φ, ∂²φ, g_μν, R_μνρσ)

L_2: Kinetic term
L_3: Cubic derivative coupling  
L_4: Quartic derivative coupling
L_5: Quintic derivative coupling
```

**Screening Mechanisms**:
- Vainshtein mechanism (strong coupling regime)
- Symmetron mechanism (environmental dependence)
- Chameleon screening (density-dependent mass)

**Tests**:
1. Can screening suppress T_μν^(eff) < 0 locally?
2. Does Vainshtein radius R_V conflict with warp bubble R?
3. QI bounds with Horndeski fluctuations

**Expected Result**: Screening works for static sources, unclear for dynamic warp bubble

### Week 5-6: DHOST Theories (If Promising)

**Goal**: Explore degenerate higher-order theories

**Only if**: Horndeski shows promise (otherwise skip to wormholes)

**Framework**:
```
DHOST extends Horndeski to higher derivatives while maintaining:
- 3 propagating DOF (no Ostrogradsky instability)
- Degenerate Hessian structure
- Well-posed Cauchy problem
```

**Tests**:
1. Beyond-Horndeski screening mechanisms
2. ANEC with DHOST modifications
3. Observational constraints (gravitational waves)

## Success Criteria

### GO Criteria (Continue to Phase C)
- ✅ At least ONE scalar-tensor model shows ANEC ≥ 0 for some warp metric
- ✅ QI violations reduced by >10⁶× (from 10²³× to <10¹⁷×)
- ✅ Screening mechanism physically realizable (not fine-tuned)
- ✅ Consistent with solar system tests (|ω_BD| > 40,000)

### NO-GO Criteria (Close FTL research entirely)
- ❌ ALL scalar-tensor models still violate ANEC
- ❌ QI gap remains >10²⁰× 
- ❌ Screening requires Planck-scale engineering
- ❌ Conflicts with gravitational wave observations

### MAYBE Criteria (Proceed to Phase C - Wormholes)
- ⚠️ Marginal improvement (ANEC violations reduced but not eliminated)
- ⚠️ QI gap narrowed to 10¹⁵-10²⁰× range
- ⚠️ Theoretical promise but extreme parameter tuning required

## Implementation Plan

### Computational Framework

**Reuse from Phase A**:
- Geodesic integrator (`integrate_geodesic.py`)
- ANEC computation (`compute_anec.py`)
- QI module (`energy_conditions/qi.py`)

**New Implementations**:
```
src/
├── scalar_field/
│   ├── brans_dicke.py          # BD field equations
│   ├── horndeski.py            # Horndeski Lagrangians
│   └── screening.py            # Screening mechanisms
├── modified_stress_energy/
│   ├── scalar_contribution.py  # T_μν^(scalar)
│   └── effective_coupling.py   # G_eff(φ)
└── metrics/
    ├── bd_alcubierre.py        # Alcubierre + BD
    ├── bd_natario.py           # Natário + BD
    └── horndeski_warp.py       # Generic Horndeski warp
```

### Validation Strategy

**Cross-checks**:
1. Solar system tests: ω_BD > 40,000 (Cassini constraint)
2. Cosmology: φ → const at late times (ΛCDM recovery)
3. GW observations: Scalar radiation < 1% (LIGO/Virgo)
4. Weak field limit: Recover GR + Yukawa correction

**Numerical Validation**:
- Energy-momentum conservation: ∇_μ T^μν_total = 0
- Scalar field equation residual < 10⁻⁸
- ANEC integration convergence (Richardson extrapolation)

## Timeline

**Week 1** (Oct 21-27):
- Brans-Dicke implementation
- Alcubierre + BD metric
- Solar system constraint validation

**Week 2** (Oct 28 - Nov 3):
- BD-ANEC computation (13 rays)
- QI bounds with scalar fluctuations
- Decision: GO/NO-GO on Brans-Dicke

**Week 3** (Nov 4-10):
- Horndeski framework implementation
- Screening mechanism tests
- Vainshtein radius analysis

**Week 4** (Nov 11-17):
- Horndeski-ANEC computation
- Screening efficacy evaluation
- Decision: GO/MAYBE/NO-GO on Horndeski

**Week 5-6** (Nov 18 - Dec 1):
- DHOST exploration (if Horndeski promising)
- OR prepare Phase C (wormholes)
- OR write formal closure (if all negative)

**Target Decision**: December 1, 2025

## Expected Outcome

**Most Likely** (70% probability):
- Scalar-tensor theories still violate ANEC
- Screening mechanisms don't work for dynamic warp bubbles
- QI gap reduced marginally (10²³× → 10²¹×, still insurmountable)
- **Result**: FTL remains impossible, proceed to Phase C (wormholes) or close

**Optimistic** (25% probability):
- Screening reduces violations significantly
- QI gap narrows to 10¹⁵-10¹⁸× (still large but promising)
- Theoretical path forward identified
- **Result**: Deeper investigation warranted (extended Phase B or Phase D)

**Breakthrough** (5% probability):
- Specific Horndeski model achieves ANEC ≥ 0
- QI violations eliminated or reduced to <10¹⁰×
- Physical realization plausible
- **Result**: FTL might be possible (revolutionary finding)

## Connection to Phase A

**Lessons from lqg-anec-framework**:
1. ✅ Production-quality metric implementations (reuse)
2. ✅ Rigorous ANEC integration (13 rays, null constraint)
3. ✅ QI framework (Ford-Roman bounds)
4. ✅ Systematic testing methodology

**New in Phase B**:
- Scalar field dynamics
- Modified stress-energy tensors
- Screening mechanism analysis
- Observational constraint validation

## Repository Structure

```
scalar-tensor-ftl-analysis/
├── README.md
├── PHASE_B_PLAN.md
├── src/
│   ├── scalar_field/
│   ├── modified_stress_energy/
│   ├── metrics/
│   └── screening/
├── tests/
│   ├── test_brans_dicke.py
│   ├── test_horndeski.py
│   └── test_screening.py
├── examples/
│   ├── demo_bd_solar_system.py
│   └── demo_screening.py
├── results/
│   ├── bd_anec_results.json
│   └── horndeski_screening.json
└── docs/
    ├── brans_dicke_derivation.md
    ├── horndeski_primer.md
    └── screening_mechanisms.md
```

## References

**Brans-Dicke**:
- Brans & Dicke (1961), Phys. Rev. 124, 925
- Will (2014), Living Rev. Relativ. 17, 4 (solar system tests)

**Horndeski**:
- Horndeski (1974), Int. J. Theor. Phys. 10, 363
- Kobayashi et al. (2011), Prog. Theor. Phys. 126, 511 (generalized Galileons)

**Screening**:
- Vainshtein (1972), Phys. Lett. B 39, 393
- Khoury & Weltman (2004), Phys. Rev. D 69, 044026 (chameleon)

**DHOST**:
- Langlois & Noui (2016), JCAP 02, 034

---

**Status**: Initialized Oct 15, 2025  
**Lead Researcher**: arcticoder  
**Expected Completion**: November-December 2025
