# 风控闸门触发流程

本文档描述 `services.risk_gate.evaluate_plan` 对交易方案执行硬风控校验的关键节点，以及闸门触发后的通知与回路。

## 流程图

```mermaid
graph TD
    A[PaperTrader 输出执行包<br/>JSON 结构化计划] --> B{调用 risk_gate.evaluate_plan}
    B --> C[拉取 risk_snapshot 与 settings
检查仓位占用、单笔敞口、止损、压力损失、当日回撤、相关性]
    C --> D{发现硬性指标超限?}
    D -- 否 --> E[返回 PASS 报告

## 代码依赖总览

- **配置层（`config/settings.py`）**：集中定义 `hard_gate_*` 上限、行情鲜度、仓位阈值等参数，允许 `.env` 覆写。`Settings` 通过 Pydantic 校验所有数值范围，是风险闸门的唯一配置入口。
- **风险快照生成（`services/risk.py::build_risk_snapshot`）**：负责聚合行情数据与账户状态，产出 `risk_snapshot`，其中包含仓位利用率、历史 VaR、情景压力测试、交叉相关性、流动性指标等。输入依赖 `market_data.fetch_price_history` 和 `risk_math` 计算器。
- **行情拉取与鲜度控制（`services/market_data.py`）**：`fetch_price_history` 在完成多级数据源回退后调用 `_ensure_freshness`，确保最新样本未超过 `market_data_max_age_minutes`。所有风险指标、流动性估计以及止损距离计算都建立在这一数据链路上。
- **量化计算器（`services/risk_math.py`）**：提供 `ScenarioShock`、`rolling_correlation`、`historical_var`、`apply_scenario` 等基础工具，为风险快照与闸门校验提供可重复的指标计算。
- **硬风控执行（`services/risk_gate.py::enforce_hard_limits`）**：按模块化方式执行仓位占用、单笔敞口、止损覆盖/方向、流动性、压力损失、当日回撤、预警上报、相关性与熔断器检查。若任何一项违反阈值即返回 `HardRiskGateReport` 携带违规详情。
- **熔断评估（`services/circuit_breaker.py::evaluate_circuit_breaker`）**：结合 `portfolio_state.risk_controls` 与最新 `risk_snapshot`，在达到连续亏损、当日亏损上限、波动率暴增或冷却期未结束时直接阻断执行，并回传需要更新的风控状态片段。
- **工作流集成（`workflows/gold_outlook.py` 等）**：在计划成型后调用 `enforce_hard_limits`，若返回 `breached=True`，立即抛出 `HardRiskBreachError` 阻断后续执行。

> 即将进行的参数调整与新逻辑必须同时更新 `settings`、`risk.py`、`risk_gate.py`，以保证配置、测度、闸门三层保持一致。
写入 outputs]
    E --> F[RiskManagerAgent 审阅并出具结论]
    F --> G[ComplianceAgent 审核并落地]
    D -- 是 --> H[返回 HardRiskBreachError
附带 breaches JSON]
    H --> I[CLI 捕获异常
写入 outputs/breaches.json]
    I --> J[ScribeAgent 记录并通知]
    J --> K[RiskManagerAgent 状态=REJECTED
列出 revision_actions]
    K --> L[HeadTraderAgent 回炉 Phase 2
按 revision_actions 改稿]
    L --> B
```

## 操作要点

1. **数据输入**：`evaluate_plan` 收到的 plan JSON 来自 PaperTrader 汇总，必须包含仓位、对冲、止损、风险测度。
2. **校验指标**：闸门读取 `config.settings.Settings` 中的 `hard_gate_*` 阈值，并结合最新 `risk_snapshot` 指标进行对比。
3. **触发结果**：
   - **通过**：返回 `RiskGateReport`; 工作流继续由 RiskManager → Compliance → Settlement。
   - **拒绝**：抛出 `HardRiskBreachError`，并在异常内嵌 `breaches` 列表（指标、实测值、上限、处置建议）。
4. **通知链路**：CLI 捕获异常后将 `outputs/risk_gate_breaches.json` 写盘；ScribeAgent 在下一轮广播给 Risk/HeadTrader。
5. **回路闭环**：RiskManagerAgent 根据异常将状态置为 `REJECTED/BLOCKED`，明确 `revision_actions`；HeadTrader 需据此修订方案并重新发起。若 `evaluated_metrics` 携带 `circuit_breaker_state_patch`，应同步写回 `portfolio_state`，以便跟踪冷却时间与连续亏损计数。
6. **数据质量护栏**：`build_risk_snapshot` 会附带 `data_quality` 元信息（实际行情来源、采样行数、鲜度分钟数），`enforce_hard_limits` 依据该信息做两件事：
    - 当行情超过 `settings.market_data_max_age_minutes` 时立即触发 `MARKET_DATA_STALE` 硬违规（遵循 `hard_gate_fail_fast` 提前退出）。
    - 根据数据源档位（如 Polygon→institutional、TwelveData→professional、YFinance→retail）自动放宽/收紧价差、滑点、最小市场深度与止损距离阈值，以避免低质量行情造成误报，同时在高质量数据下保持原有严格度。

> 若需扩展新指标，请在 `services/risk_gate.py` 新增校验函数，并在 `tests/test_risk_gate.py` 添加覆盖。
