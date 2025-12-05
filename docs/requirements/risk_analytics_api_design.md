# Risk Analytics API Design

_Date: 2025-12-05_

## 1. Objective

Provide a consistent Python API for reusable risk analytics so every agent and
tooling component depends on the same validated computations.  The interface
must be:

- Deterministic and side-effect free (pure functions returning dictionaries or
  dataclasses).
- Serializable into existing workflow payloads (JSON-friendly results).
- Guarded by unit tests with realistic edge-case fixtures.
- Efficient enough to run inside synchronous agent steps (< 200 ms for 5-year
  daily histories, < 1 s for 1-minute data windows).

## 2. Scope

Metrics covered in this iteration:

1. **Tail Risk**
   - Historical VaR / Expected Shortfall (ES) using configurable confidence
     levels and lookback periods.
   - Parametric (Gaussian / Cornish-Fisher) variants for comparison.
2. **Scenario Stress Tests**
   - Deterministic price/volatility shocks applied to price series or PnL
     profiles.
   - Composite macro scenarios (e.g., yield curve shift + FX move).
3. **Liquidity Diagnostics**
   - Order book depth estimation, implied slippage, participation rate limits.
   - Session-aware adjustments (reuse settings-driven relaxation factors).
4. **Regime Detection**
   - Rolling volatility classification (calm / normal / stressed).
   - Correlation clustering vs. reference assets.

Out of scope: Options Greeks, Monte-Carlo path generation, cross-asset
portfolio optimization (track for later phases).

## 3. Module Layout

```
src/autogentest1/services/risk_analytics/
    __init__.py
    tail_risk.py          # VaR / ES calculators
    stress.py             # Scenario application utilities
    liquidity.py          # Liquidity & slippage estimators
    regimes.py            # Regime detection helpers
    schemas.py            # Dataclasses / TypedDict definitions
```

Each module exposes top-level functions returning dataclasses defined in
`schemas.py`.  These dataclasses must implement a `.to_dict()` helper used by
agents when serializing payloads.

## 4. Core APIs

### 4.1 Tail Risk

```python
@dataclass(frozen=True)
class TailRiskResult:
    returns_window: int
    confidence: float
    historical_var: float
    historical_es: float
    parametric_var: float | None
    parametric_es: float | None
    observations: int


def compute_tail_risk(
    returns: pd.Series,
    *,
    confidence: float = 0.99,
    lookback: int | None = None,
    method: Literal["gaussian", "cornish"] | None = "gaussian",
) -> TailRiskResult:
    ...
```

- `lookback=None` => use entire series; otherwise roll the last *n* points.
- `method=None` skips parametric measures for performance-sensitive paths.

### 4.2 Scenario Stress

```python
@dataclass(frozen=True)
class ScenarioResult:
    base_price: float
    base_q: float | None          # optional quantity exposure
    outcomes: list[dict[str, float]]  # e.g. [{"label": "minus_5pct", "price": 1800.0, "pnl": -2.3}]


def apply_price_scenarios(
    prices: pd.Series,
    shocks: Sequence[ScenarioShock],
    *,
    exposure: float | None = None,
    transaction_cost_bps: float = 0.0,
) -> ScenarioResult:
    ...
```

- `ScenarioShock` reused from `services.risk_math` for compatibility.
- When `exposure` provided, compute projected PnL under each scenario.

### 4.3 Liquidity Diagnostics

```python
@dataclass(frozen=True)
class LiquiditySnapshot:
    avg_volume: float | None
    latest_volume: float | None
    spread_bps: float | None
    atr_bps: float | None
    implied_depth_oz: float | None
    required_depth_oz: float | None
    slippage_bps: float | None
    session: str
    relaxation_factor: float
    warnings: list[str]


def evaluate_liquidity(
    *,
    metrics: Mapping[str, Any],
    order_size_oz: float,
    settings: Settings,
    session_key: str,
) -> LiquiditySnapshot:
    ...
```

- `metrics` compatible with output from `risk.py::_compute_liquidity_metrics`.
- Reuses existing calibration maps while centralizing formulae.

### 4.4 Regime Detection

```python
@dataclass(frozen=True)
class RegimeClassification:
    window: int
    volatility: float | None
    regime: Literal["calm", "normal", "stressed"]
    z_score: float | None
    reference_assets: dict[str, float]  # correlation coefficients


def classify_regime(
    returns: pd.Series,
    *,
    window: int = 60,
    thresholds: Mapping[str, float] | None = None,
    reference_series: Mapping[str, pd.Series] | None = None,
) -> RegimeClassification:
    ...
```

- Default thresholds: calm < 0.5 * baseline vol, stressed > 1.5 * baseline.
- `reference_series` optional; when supplied, compute correlations and include
  top contributors in result.

## 5. Serialization Contract

All result dataclasses must implement:

```python
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
```

This keeps them JSON-safe for agents and audit logs.  Export these dataclasses
through `services.risk_analytics.__all__` for direct imports.

## 6. Testing Strategy

- Unit tests per module covering:
  - Empty/NaN series handling.
  - Degenerate cases (single observation, zero volatility).
  - Known-value fixtures compared against NumPy/Pandas calculations.
- Integration tests invoking the primary entry points with realistic gold
  price slices (~750 daily points) to assert run time and output ranges.
- Property-based tests (e.g., via Hypothesis) for regime classification to
  ensure monotonic threshold logic.

## 7. Migration Plan

1. Land module scaffolding and dataclasses (current sprint).
2. Update `services/risk.py` / `services/risk_gate.py` to call the new helpers
   instead of bespoke inline calculations.
3. Adjust agent prompts (`RiskManagerAgent`, `ComplianceAgent`) to cite the
   new metrics where available.
4. Deprecate redundant utilities in `services/risk_math.py` once parity tests
   confirm behavior.

## 8. Open Questions

- Should tail-risk functions support intraday (minute) sampling with automatic
  resampling? _Decision: track separately; current API assumes pre-aggregated
  returns._
- What is the canonical baseline volatility for regime detection? _Proposal:
  derive from `portfolio_state.risk_controls.baseline_vol_annualized` with
  fallback to trailing 60-day realized volatility._
- Do we need streaming/incremental updates? _Not in this phase; batch
  recomputation per workflow run is acceptable._

## 9. Next Actions

- [ ] Implement module skeletons with Stub return values.
- [ ] Draft test fixtures using recorded gold price series (see `tests/fixtures`).
- [ ] Schedule design review with veteran advisor for parameter thresholds.
- [ ] Create tracking issues for follow-up implementation tasks.
