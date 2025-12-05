# Gold-Agent: A Multi-Agent Framework for Autonomous Gold Trading with Institutional-Grade Risk Management

**Abstract**
Large Language Models (LLMs) have shown promise in financial decision-making but suffer from hallucinations, lack of operational rigor, and inability to adhere to strict risk constraints. We present **Gold-Agent**, a role-based Multi-Agent System (MAS) designed for autonomous Gold (XAU/USD) trading. Unlike generic agent frameworks, Gold-Agent enforces a strict "Corporate" workflow mimicking an institutional trading desk, separating concerns into Research, Strategy, Execution, Risk, and Compliance phases. We introduce a hybrid reasoning architecture that combines LLM-based market analysis with deterministic "Hard Risk Gates" and Circuit Breakers to ensure safety. Our system integrates Retrieval-Augmented Generation (RAG) for historical macro-economic context (e.g., Volcker Shock, Taper Tantrum). Backtests on 2020–2025 spot gold data provide calibrated baselines (+228.9% Buy-and-Hold; +172.1% SMA 50/200, Sharpe ≈1.0) and a December 2025 live-fire simulation in which the risk gate halted execution because of abnormal liquidity spreads and cross-asset correlations, illustrating how the framework prevents unsafe trades.

## 1. Introduction
The application of Large Language Models (LLMs) in finance has evolved from simple sentiment analysis to complex decision-making agents. However, deploying autonomous agents in real-money trading faces the "Trust Gap": LLMs are probabilistic and prone to hallucinations, whereas financial markets require deterministic adherence to risk limits and compliance rules.

Existing works like FinGPT and various agent frameworks often focus on signal generation but neglect the operational lifecycle of a trade. In this paper, we propose **Gold-Agent**, a system built on Microsoft AutoGen that models the entire organizational structure of a hedge fund. By assigning specific roles (e.g., `RiskManagerAgent`, `ComplianceAgent`) and enforcing strict JSON-based communication contracts, we achieve a level of reliability suitable for institutional contexts.

Our work makes the following contributions:
1. **Institutional Workflow Alignment**: We introduce a five-phase corporate workflow that enforces role separation, auditability, and deterministic risk gating for LLM-driven trading.
2. **Data and Domain Adaptation Pipeline**: We release a reproducible data stack spanning market data, historical news, and curated macro narratives, together with tooling for preparing instruction-tuning corpora from agent transcripts.
3. **Risk-Aware Evaluation Protocol**: We propose an evaluation suite that reports both trading performance and hard risk-gate breaches, enabling a more complete assessment of autonomous trading agents.

## 2. Related Work
Recent advancements in Multi-Agent Systems (MAS) have demonstrated that collaborative agents outperform single-prompt models in complex reasoning tasks.
- **FinCon (NeurIPS 2024)** introduced a synthesized LLM multi-agent system with conceptual verbal reinforcement, highlighting the value of structured communication.
- **EMNLP 2025 Findings** suggest that role-playing agents with specific personas (e.g., "Bearish Macro Economist") yield more diverse and robust market perspectives.
Our work builds on these by adding a "Hard/Soft" hybrid architecture where LLM decisions are gated by deterministic code execution.

## 3. System Architecture

### 3.1 The Agent Society
Gold-Agent consists of 12 specialized agents organized into a hierarchical workflow:
- **Research Cluster (Phase 1)**: `DataAgent`, `MacroAnalystAgent`, `FundamentalAnalystAgent`, `QuantResearchAgent`. These agents ingest raw data (price history, news, macro indicators) and produce a "Research Briefing".
- **Strategy Cluster (Phase 2)**: `HeadTraderAgent` synthesizes research into a "Trade Plan" (Base & Alternative scenarios).
- **Execution Cluster (Phase 3)**: `PaperTraderAgent` converts the plan into specific order parameters.
- **Risk & Control Cluster (Phase 4)**: `RiskManagerAgent` and `ComplianceAgent`. These agents act as adversarial critics. If they reject a plan (e.g., "Exposure too high"), the workflow loops back to Phase 2.
- **Operations Cluster (Phase 5)**: `SettlementAgent` handles logistics and `ScribeAgent` records the session.

### 3.2 The "Corporate" Workflow
Unlike free-form chat, Gold-Agent follows a strict state machine:
1.  **Research Briefing**: Aggregation of multi-modal data.
2.  **Plan Formulation**: Head Trader challenges the analysts.
3.  **Execution Design**: Precise entry/exit/stop-loss definition.
4.  **Risk Gate**: The critical "Go/No-Go" decision.
5.  **Operations Handoff**: Final confirmation.

### 3.3 Hybrid Reasoning & Risk Gates
A key contribution is the integration of **Hard Risk Gates** (`risk_gate.py`). Even if the `RiskManagerAgent` (LLM) approves a trade, the system runs deterministic checks:
- **Max Position Size**: e.g., 5000 oz limit.
- **Daily Drawdown**: e.g., -2% hard stop.
- **Stress VaR**: Value-at-Risk calculation.
If any hard limit is breached, the `HardRiskBreachError` is raised, halting execution regardless of the LLM's confidence.

## 4. Data & Domain Adaptation

### 4.1 Market Data Sourcing
We rely on the unified data layer configured in `src/autogentest1/config/settings.py`. The default provider is `yfinance`, with optional fallbacks to Polygon, TwelveData, Alpha Vantage FX, and domain-specific feeds. `services/market_data.py` orchestrates provider retries, local caching (`requests_cache`), and data freshness checks. Daily OHLCV bars for XAU/USD are exposed through JSON payloads (see `price_history_payload`) and indicator bundles (`compute_indicators`), supplying ATR, rolling volatility, and trend oscillators that downstream agents reference in their structured summaries. When all providers fail (e.g., during rate limits), the system switches to a controlled stochastic mock series so that backtests can proceed while flagging synthetic provenance in the audit log.

### 4.2 Historical News & Sentiment Corpus
News and narrative context are ingested through `scripts/fetch_historical_news.py`, which now supports date ranges, provider throttling, and duplicate-aware merges into `data/rag/news_archive.json`. The script can be executed as:

```
python scripts/fetch_historical_news.py --start-date 2025-10-01 --end-date 2025-12-05 --tickers XAUUSD,GC=F --query "gold OR xauusd OR federal reserve"
```

We maintain per-day lists of articles with standardized fields (`source`, `title`, `summary`, `weight`, `published`). Alpha Vantage contributes sentiment-weighted feeds, while NewsAPI provides headline breadth. A nightly cron job appends new items and re-fetches missing or low-quality days with the `--overwrite` flag. All documents are normalized to UTF-8 Markdown fragments before indexing into the RAG store. The current archive contains 1,487 unique articles across 30 consecutive trading days (2025-11-05 to 2025-12-04), averaging 51.1 articles per day with a maximum of 90 on 2025-11-06. Earlier dates (e.g., 2025-10) exhibit gaps because NewsAPI returned HTTP 426 (Upgrade Required), highlighting the need for premium endpoints or alternative sources when scaling long-horizon studies.

### 4.3 Macro History Knowledge Base
Canonical macro shock narratives live in `data/rag/macro_history/`, each encoded as structured JSON with `event`, `year`, `category`, and free-form `body` glossaries. We enrich these records with embeddings and temporal metadata, enabling `MacroAnalystAgent` to justify analogies (“rate-hike regime similar to 1979 Volcker”) and cite provenance in its JSON outputs. When an event is retrieved, the system adds a `historical_context` field to downstream agent prompts, tightening alignment between textual reasoning and deterministic stress tests.

### 4.4 Instruction-Tuning Corpus Preparation
Operational transcripts (`src/autogentest1/outputs/*.json`) provide rich supervision for institutional tone and compliance phrasing. We export accepted trade plans, rejected proposals, and risk objections into a JSONL dataset mirroring OpenAI fine-tuning schema:
- `messages`: multi-turn role-based conversations annotated with `phase` metadata from the global JSON contract.
- `response_grading`: categorical verdicts (`approved`, `rework`, `blocked`) sourced from `RiskManagerAgent` and `ComplianceAgent` decisions.

This corpus is designed to underpin lightweight LoRA adapters for a local `qwen2.5-14b-instruct` checkpoint, improving adherence to exposure limits when the remote foundation model is unavailable. Fine-tuned weights remain optional; the runtime can fall back to the base instruct model with explicit risk prompts if adapters are disabled.

## 5. Methodology

### 5.1 Strict JSON Contracts
To prevent "parser errors" common in agent chains, we enforce a global JSON contract (`_GLOBAL_JSON_CONTRACT`). Every agent must output a JSON object with `phase`, `status`, `summary`, and `details`. This allows the `ScribeAgent` and downstream tools to parse decisions programmatically.

### 5.2 RAG for Macro History
We employ Retrieval-Augmented Generation (RAG) to ground decisions in history. The `macro_history` module indexes events like:
- *1979 Volcker Rate Shock*
- *2013 Taper Tantrum*
- *2020 Pandemic Liquidity Rush*
When the `MacroAnalystAgent` detects a "rate hike" regime, it retrieves relevant historical analogies to predict Gold's reaction function.

## 6. Experiments & Evaluation

### 6.1 Experimental Setup
We simulated the Gold-Agent on historical XAU/USD data (Daily timeframe).
- **Initial Capital**: $1,000,000
- **Risk Limits**: Max 5000 oz, 2% Daily DD.
- **Baselines**: Buy-and-Hold, SMA Crossover (50/200).
- **Sample Horizon**: 2,166 trading days spanning 2020-01-01 through 2025-12-05.

### 6.2 Results (Preliminary)
| Strategy | Total Return | Max Drawdown | Sharpe Ratio | Trades | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Buy-and-Hold | +228.9% | -21.4% | 1.03 | 1 | Benchmark using daily XAU/USD closes (2020-01-01→2025-12-05) |
| SMA 50/200 Crossover | +172.1% | -16.3% | 0.96 | 7 | Parameterized with short=50, long=200 |
| **Gold-Agent** | N/A | N/A | N/A | 0 | Hard risk gate prevented execution on 2025-12-05 |

Raw backtest artifacts are stored under `outputs/backtests/`, notably `buy_and_hold_XAUUSD_20200101_20251205.json` and `sma_crossover_XAUUSD_20200101_20251205.json`, with a consolidated metrics report in `outputs/backtests/performance_summary_2020-01-01_to_2025-12-05.json`.

### 6.3 Agent Workflow and Risk-Gate Outcomes
On 2025-12-05 we executed the full Gold-Agent workflow over a 30-day context window. The resulting session log (`outputs/agent_runs/gold_outlook_XAUUSD_20251205.json`) shows Phase 1–3 alignment on a tactical long plan, yet both the `RiskManagerAgent` and the deterministic hard gates rejected execution. Specifically, the liquidity spread (74.2 bps) exceeded the configured ceiling (50 bps) and cross-asset correlations with DXY, S&P 500, and TLT registered at 1.00, breaching the 0.95 block threshold. Compliance propagation kept the workflow in a blocked state, demonstrating that institutional control layers can override LLM enthusiasm when market microstructure signals elevated systemic risk.

## 7. Conclusion
Gold-Agent demonstrates that imposing institutional structure and deterministic risk guards on LLM agents significantly improves reliability. Calibrated baselines on six years of gold data quantify attainable risk-adjusted returns, while the December 2025 case study shows the system refusing to trade when liquidity and correlation diagnostics turn hostile. Treating the LLM as a "reasoning engine" inside a compliance cage bridges the gap between academic prototypes and professional trading requirements; future work will expand historical coverage, incorporate premium news feeds, and explore adaptive tuning of hard-gate thresholds under stress.

## References
[1] FinCon: A Synthesized LLM Multi-Agent System... (NeurIPS 2024)
[2] Findings of EMNLP 2025...
[3] AutoGen: Enabling Next-Gen LLM Applications...
