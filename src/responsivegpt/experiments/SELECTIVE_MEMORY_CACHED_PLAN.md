# Selective Memory-Augmented Fast-Slow ResponsiveGPT Plan

## Objective

Reduce timeout, token, and wall-clock cost while preserving the paper's core contribution:
fast reactive reasoning, phase-level slow planning, grounded RAG evidence, and adaptive driver profiles for high-risk strong-interaction corner cases.

## Final Architecture

1. Deterministic safety scan
   - Compute TTC, THW, DRAC, DCPA, future distance, physical risk index, and interaction metadata for every frame.
   - Obvious safe or dangerous states can be handled by deterministic fast rules in later phases.

2. Scenario-conditioned RAG
   - Use dataset and interaction metadata to retrieve relevant law, case, and scenario evidence.
   - Cache evidence packs by scene signature, RAG mode, top-k, driver type, feedback, and planning hint.

3. Phase-level slow planning
   - Planning stays as the slow-thinking module.
   - Planning output is cached by the complete planning prompt and reused when the risk phase/context is unchanged.

4. Selective invocation
   - LLM is invoked only when the state is critical, novel, uncertain, evidence-changing, or planning/reactive-conflicting.
   - Exact LLM responses are cached by prompt/model/schema configuration.

5. Causal case memory
   - Later phase. Store only system-generated experience from earlier run-order cases.
   - Never store ground-truth labels, future outcomes, or correctness.
   - Direct reuse is allowed only under strict similarity, evidence-overlap, and risk-consistency gates.

6. Budget governor
   - Later phase. Dynamically tighten thresholds when request, token, or wall-time budgets are close to exhaustion.

## Implementation Phases

### Phase 1: Cache Foundation

Implemented first because it is low-risk and does not change the method's decision logic.

- Exact LLM JSON cache.
- RAG evidence-pack cache.
- Planning output cache.
- Summary/trace metrics:
  - `llm_cache_hits`
  - `llm_cache_misses`
  - `planning_cache_hits`
  - `planning_cache_misses`
  - `rag_cache_hits`
  - `rag_cache_misses`
  - cache hit rates in `token_time_efficiency`

### Phase 2: Phase-Level Planning

- Add risk-phase segmentation: approaching, conflict, recovery.
- Reuse planning hints inside a stable phase.
- Refresh planning only when risk phase changes, evidence changes, or reactive decisions conflict with planning.

### Phase 3: Selective Invocation Model

- Replace pure stride-based gating with a unified novelty/uncertainty/conflict gate.
- Inputs:
  - physical risk metrics
  - risk deltas
  - RAG evidence overlap
  - planning age/conflict
  - memory similarity
  - profile type
- Outputs:
  - reuse last decision
  - use case memory prior
  - call reactive LLM

### Phase 4: Causal Case Memory

- Build a memory bank with strict causal filtering:
  - `source_run_order < current_run_order`
  - no ground-truth labels
  - no future outcomes
- Memory record:
  - scene signature
  - physical metrics
  - interaction type
  - profile name
  - evidence ids
  - planning hint
  - generated decision
  - confidence
- Modes:
  - `off`
  - `calibration_only`
  - `causal_online_prior`
  - `causal_online_reuse`

### Phase 5: Budget Governor

- Track request count, token estimate, cache hit rate, wall time, and remaining budget.
- Tighten gate when nearing budget:
  - increase uncertainty threshold
  - reduce RAG top-k
  - extend stale window
  - prefer planning cache and memory prior
  - reserve LLM for critical frames

## Evaluation Plan

1. Main performance matrix:
   - no RAG
   - naive RAG
   - full grounded RAG
   - all with planning on

2. Efficiency ablation:
   - baseline
   - + exact LLM cache
   - + RAG cache
   - + planning cache
   - + selective invocation
   - + causal memory
   - + budget governor

3. Safety and leakage audit:
   - memory source run order
   - source split
   - false reuse rate
   - evidence-overlap distribution
   - held-out frozen-memory evaluation

4. Reporting metrics:
   - F1, precision, recall, alignment accuracy
   - grounded decision rate
   - hallucinated citation rate
   - planning/reactive consistency
   - LLM attempts per frame
   - LLM decisions per frame
   - planning attempts per event
   - cache hit rates
   - wall-clock time
   - token usage
