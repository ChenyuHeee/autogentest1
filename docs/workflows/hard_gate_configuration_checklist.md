# Hard Gate Configuration Checklist

Use this short checklist before promoting a new release or refreshing
infrastructure so the hard risk gate continues to match desk policy.

## Environment Variables to Review

- `HARD_GATE_MIN_STOP_DISTANCE_PCT` / `HARD_GATE_MAX_STOP_DISTANCE_PCT`
  - Defaults: `0.5` / `3.0`
  - Confirm strategy-specific stop ranges (e.g., wider for swing trading).
- `HARD_GATE_STOP_COVERAGE_RATIO`
  - Default: `1.0`
  - Review exemptions supplied via `HARD_GATE_STOP_COVERAGE_EXEMPTIONS`.
- `HARD_GATE_MAX_POSITION_UTILIZATION_ROUTINE`
  - Default: `0.6` (with `0.8` elevated, `0.95` hard cap).
  - Adjust per prevailing leverage policy.
- `HARD_GATE_INCREMENTAL_POSITION_UTILIZATION_LIMIT`
  - Default: `0.3` (30% of max position per trade).
- `HARD_GATE_MAX_SINGLE_ORDER_OZ`
  - Default: `2000`.
  - Confirm alignment with compliance limits and liquidity conditions.
- `HARD_GATE_MIN_LIQUIDITY_DEPTH_OZ`
  - Default: `800`.
  - Increase when trading outside top-tier liquidity windows.
- `HARD_GATE_MAX_SPREAD_BPS` / `HARD_GATE_MAX_SLIPPAGE_BPS`
  - Defaults: `40` / `35`.
  - Consider session-specific adjustments where necessary.
- `HARD_GATE_CORRELATION_WARNING_THRESHOLD` / `HARD_GATE_CORRELATION_LIMIT_THRESHOLD`
  - Defaults: `0.6` / `0.8` (block threshold `0.95`).
  - Tune according to basket concentration policy.
- `HARD_GATE_LIQUIDITY_SESSION_RELAXATION`
  - Default map: `{asia: 1.5, london: 1.2, newyork: 1.0, off_hours: 1.3}`.
  - Update when trading hours or liquidity assumptions change.
- `MARKET_DATA_MAX_AGE_MINUTES`
  - Default: `30` minutes.
  - Ensure upstream feeds guarantee fresher data or tighten accordingly.

## Recommended Actions Before Deployment

1. Export prospective environment variables (`poetry run python -m autogentest1.config.dump_settings`).
2. Compare overrides with the defaults listed above.
3. Capture approvals for any deviation from the trading policy.
4. Audit the deployed environment after rollout to verify the values
   match the approved configuration.
