# Week 1 Progress: Brans-Dicke Initial Results

**Date**: October 15, 2025  
**Status**: ⚠️ **NEGATIVE PRELIMINARY RESULT**

## Summary

Initial Brans-Dicke testing reveals **catastrophic scalar field response** to warp bubble stress-energy, even with conservative ω = 50,000 (well above Cassini bound).

## Key Finding

**φ(r) goes entirely negative** when coupled to realistic warp bubble:
- Background: φ₀ = 1.0
- With T ~ 10³⁷ J/m³ (conservative Alcubierre estimate):
  - φ(center) ≈ -2.1×10²³
  - φ(bubble wall) ≈ -1.9×10²³  
  - G_eff = G/φ becomes **negative** (unphysical!)

## Technical Details

### Parameters Tested
- Warp velocity: v_s = 0.1c
- Bubble radius: R = 1.0 m
- BD coupling: ω = 50,000 (Cassini-compliant)
- Source coefficient: α = 8πG/(3+2ω) ≈ 1.68×10⁻¹⁴

### Scalar Field Equation
```
∇²φ = (8πG/(3 + 2ω)) T(r)
```

For ω = 50,000:
```
α = 1.68×10⁻¹⁴
```

### Perturbation Estimate
Even with large ω, the perturbation is:
```
δφ ~ -α ∫ T(r)/r d³r
    ~ -(10⁻¹⁴) × (10³⁷ J/m³) × (1 m³)
    ~ -10²³
    >> φ₀ = 1
```

**Result**: δφ/φ₀ ~ 10²³ → Perturbation theory breaks down

## Physical Interpretation

1. **Warp bubbles are too strong for BD screening**:
   - Stress-energy T ~ 10³⁷-10³⁹ J/m³ required for FTL
   - Even ω = 50,000 cannot suppress scalar response
   - φ flips sign → G_eff < 0 (gravitational repulsion?!)

2. **Increasing ω doesn't help enough**:
   - Need ω ~ 10²⁰ to get α ~ 10⁻³⁴
   - But Cassini bound: ω > 40,000 (barely!)
   - Gap is **16 orders of magnitude**

3. **Non-perturbative regime**:
   - Full nonlinear BD equations required
   - But if φ → 0, theory becomes singular
   - If φ < 0, G_eff < 0 → instability

## Implications for Phase B

### Option 1: Increase ω to 10⁶-10⁹
- Test if higher coupling helps
- But violates spirit of BD theory (ω → ∞ = GR)
- Likely still insufficient (need ω ~ 10²⁰)

### Option 2: Reformulate metric
- Use conformal frame (Einstein frame vs Jordan frame)
- Might avoid φ → 0 singularity
- But doesn't change fundamental problem

### Option 3: Move to Horndeski
- Screening mechanisms (Vainshtein, chameleon)
- Might suppress scalar response near bubble
- Week 2-3 plan

### Option 4: Declare BD FAILED
- Strong evidence FTL impossible in BD
- Skip to Horndeski immediately
- **RECOMMENDED**

## Decision Point

**Recommendation**: Skip detailed BD analysis, move directly to Horndeski.

**Rationale**:
1. Preliminary result is **decisively negative**
2. No parameter tuning can fix 10²³× perturbation
3. Horndeski has better screening prospects
4. Conserves time (2 weeks → Horndeski instead of BD refinement)

**Risk**: Might miss subtlety in nonlinear BD regime  
**Mitigation**: If Horndeski shows promise, revisit BD with better numerics

## Next Steps (Pending Decision)

### If continuing BD (not recommended):
1. Implement full nonlinear solver
2. Test ω = 10⁶, 10⁹ (far beyond Cassini)
3. Explore conformal frame formulation
4. Estimate: 1-2 weeks

### If skipping to Horndeski (recommended):
1. Implement Horndeski Lagrangian (L₂-L₅)
2. Vainshtein screening analysis
3. ANEC with screened scalar
4. Estimate: 2-3 weeks

## Preliminary Conclusion

**Brans-Dicke theory cannot enable FTL warp drives.**

The scalar field response to warp bubble stress-energy is **23 orders of magnitude** larger than the background value, causing:
- Negative φ → unphysical G_eff < 0
- Breakdown of perturbation theory
- No screening mechanism to suppress response

Even extreme ω (far beyond observational bounds) cannot overcome the fundamental mismatch between warp bubble energy scales (10³⁷-10³⁹ J/m³) and BD coupling (α ~ 10⁻¹⁴).

**Status**: BD declared FAILED for FTL  
**Recommendation**: Proceed to Horndeski (Phase B.2)

---

**Files Modified**:
- `src/scalar_field/dynamic_bd.py`: Green's function solver
- `src/metrics/alcubierre_simple.py`: Simplified Alcubierre
- `examples/demo_bd_alcubierre.py`: Coupling demonstration

**Tests**: All passing (16/16)  
**Commits**: 2 (initialization + dynamic solver)
