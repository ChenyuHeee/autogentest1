# 回测参数扫描工作流

本文档记录 `services.backtest_suite` 提供的参数扫描工具链，以及如何在日常研究流程中使用它来校准策略参数。

## 功能概览

- **单次历史载入**：复用一份行情数据，对多个参数组合执行 `services.backtest.run_backtest`，避免重复抓取。
- **指标打分**：默认以夏普率作为评估指标，可切换为任意回测输出中的数值型指标（如 `total_return`、`max_drawdown`）。
- **兜底逻辑**：当主评估指标缺失或为 `NaN` 时，会自动回退到 `total_return` 作为得分，避免因指标为空导致扫描中断。
- **失败收集**：对参数无效、触发回测错误的组合进行归档，便于后续分析（例如短周期大于长周期的 SMA 情况）。
- **排名与摘要**：输出包含所有成功组合的排名、最佳配置、得分概览以及历史数据区间信息。

## 核心接口

```python
from autogentest1.services.backtest_suite import fetch_and_run_parameter_sweep

result = fetch_and_run_parameter_sweep(
    symbol="XAUUSD",
    days=730,
    strategy="sma_crossover",
    parameter_grid={
        "short_window": [10, 20, 30],
        "long_window": [50, 75, 100],
    },
    evaluation_metric="sharpe_ratio",
    fallback_metric="total_return",
    top_n=5,
)
```

返回结构示例（字段略）：

- `summary`: 统计信息，包含成功/失败次数、最佳参数、历史数据范围等。
- `best`: 排名第一的参数组合，带回测完整结果（含仓位曲线、交易记录、指标集合）。
- `runs`: 全量参数组合的打平结果，成功项附带得分、指标、回测原始输出；失败项包含错误信息。
- `top`: 前 `top_n` 名的精简视图，方便在 UI 或日志中快速展示。

## 使用建议

1. **参数边界先验**：在设置网格前，先根据策略原理确定合理区间，避免大规模无效组合拖慢扫描。
2. **评估指标选择**：
   - 夏普率适合波动稳定的策略。
   - `total_return` 更直观，适合初期粗筛。
   - `max_drawdown` 建议配合 `metric_goal="maximize"`（因为 drawdown 为负且越接近 0 越好）。
3. **运行成本控制**：`top_n` 可限制需要保留完整回测结果的数量，减少输出体积。
4. **后续衔接**：最佳参数可直接写回策略配置，或联动结构化审计日志记录调参过程。

## 与工具代理集成

`tools.backtest_tools.run_parameter_sweep` 封装了同样的逻辑，供 AutoGen 工具代理调用：

```python
from autogentest1.tools import run_parameter_sweep

report = run_parameter_sweep(
    symbol="GC=F",
    strategy="sma_crossover",
    parameter_grid={"short_window": [5, 10], "long_window": [20, 30]},
)
```

当行情抓取失败时会返回 `{"error": ...}`，可在代理层做友好提示或重试。

## 下一步

- 将回测输出与风险闸门、熔断逻辑联动，统计硬风控触发频次。
- 支持自定义评估函数（例如目标收益-回撤比）。
- 提供批量导出能力（CSV/Parquet），便于量化研究人员做进一步分析。
