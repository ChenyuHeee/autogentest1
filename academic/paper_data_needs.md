# 论文写作数据需求清单 (Paper Data Requirements)

本文档详细列出了完成 `docs/academic_paper_draft.md` 所需的具体数据支持。为了确保论文的科学性和可复现性，我们需要准备以下几类数据。

## 1. 市场行情数据 (Market Data)
**用于章节**: `6.1 Experimental Setup`, `6.2 Results`

*   **标的**: 现货黄金 (XAU/USD)
*   **时间范围**: 建议覆盖至少一个完整的市场周期（例如：2020年1月1日 - 2025年12月31日），或者针对特定事件的窗口期（如2025年11月）。
*   **频率**: 日线 (Daily) - 论文中已明确提及。
    *   *可选*: 分钟级数据 (用于更精细的滑点模拟，虽非必须但能增加论文深度)。
*   **字段**: `Date`, `Open`, `High`, `Low`, `Close`, `Volume`.
*   **用途**:
    *   计算基准策略 (Buy-and-Hold, SMA Crossover) 的回报。
    *   作为智能体回测环境的“环境状态”。

## 2. 宏观历史与 RAG 知识库 (Macro History & RAG Corpus)
**用于章节**: `3.3 Hybrid Reasoning`, `5.2 RAG for Macro History`

*   **核心事件数据**:
    *   **1979 Volcker Rate Shock**: 利率急剧上升期的黄金表现、通胀数据、美联储会议纪要摘要。
    *   **2013 Taper Tantrum**: 缩减恐慌期间的市场新闻、美债收益率变化、黄金抛售潮描述。
    *   **2020 Pandemic Liquidity Rush**: 疫情初期的流动性危机、央行救市政策、黄金先跌后涨的逻辑。
*   **数据格式**: 结构化 JSON 或 Markdown 文档（目前已存在于 `data/rag/macro_history/`，需确认内容完整性）。
*   **验证点**: 需要展示 `MacroAnalystAgent` 如何成功检索到这些案例并将其应用于当前的决策逻辑（例如：“当前通胀类似1979年，建议做多”）。

## 3. 历史新闻与舆情数据 (Historical News & Sentiment)
**用于章节**: `3.1 The Agent Society`, `6.2 Results` (隐含在 Gold-Agent 的超额收益中)

*   **数据源**: 真实的历史新闻存档（如 Alpha Vantage, NewsAPI 抓取的数据）。
*   **覆盖范围**: 必须与**市场行情数据**的回测时间段严格对应。
    *   *现状*: `data/rag/news_archive.json` 已包含 2024年1月的样例数据和 2025年11月6日 的真实抓取，但仍存在大量日期空缺。
    *   *需求*: 使用新版 `scripts/fetch_historical_news.py` 批量抓取至少 **一个月**（建议 2025年10月1日 - 2025年12月5日） 的新闻数据，并对缺失日期补充。
*   **字段**: `Title`, `Summary`, `Source`, `PublishedTime`, `URL`.
*   **衍生数据**:
    *   **Sentiment Score**: 每篇新闻的情绪打分 (-1.0 到 1.0)。
    *   **Topic Tags**: 新闻主题（如 "Inflation", "Geopolitics", "Fed Policy"）。

## 4. 实验结果数据 (Experimental Results)
**用于章节**: `6.2 Results`

为了填充论文中的对比表格，我们需要运行以下三个实验并记录详细日志：

### A. 基准策略 (Baselines)
1.  **Buy-and-Hold**:
    *   全周期持有收益率。
    *   最大回撤 (Max Drawdown)。
2.  **SMA Crossover (50/200)**:
    *   交易信号记录 (金叉买入/死叉卖出)。
    *   净值曲线 (Equity Curve)。

### B. Gold-Agent (本系统)
*   **交易日志 (Trade Log)**:
    *   每一笔交易的 `Entry Date`, `Exit Date`, `Entry Price`, `Exit Price`, `PnL`, `Reasoning` (关键：Agent 为什么做这笔交易？是基于宏观还是技术面？)。
*   **风险指标**:
    *   **Risk Breaches**: 记录被 `RiskManagerAgent` 或 `Hard Risk Gate` 拒绝的交易次数（用于证明 "0 Risk Breaches" 的安全性）。
    *   **Sharpe Ratio**: 基于日收益率序列计算。
*   **消融实验 (Ablation Study - 可选但加分)**:
    *   *无 RAG 版本*: 只有技术分析 Agent 的表现。
    *   *无 Risk Gate 版本*: 去掉风控后的表现（展示虽然收益可能高，但回撤巨大）。

## 5. 系统运行指标 (System Metrics)
**用于章节**: `7. Conclusion` 或 `Discussion`

*   **Token 消耗**: 运行一次完整回测消耗的 Token 数量（评估成本效益）。
*   **延迟 (Latency)**: 从输入市场数据到生成最终订单的平均耗时（评估实时交易的可行性）。
*   **幻觉率 (Hallucination Rate)**: (定性分析) 记录 Agent 编造数据或引用不存在新闻的案例（如有）。

## 6. 指令微调语料 (Instruction-Tuning Corpus)
**用于章节**: `4.4 Instruction-Tuning Corpus Preparation`

*   **数据源**: `src/autogentest1/outputs/` 下的多轮对话与 `~/.autogentest1/audit/` 目录中的风险评审日志。
*   **格式要求**: 整理为 JSONL，每条样本包含 `messages`（多角色对话，附 `phase`）与 `response_grading` 标签。
*   **规模目标**: 至少 1,000 条高质量实例，覆盖通过、驳回、补充信息三类结论。
*   **预处理**: 去除包含敏感 API 密钥或客户信息的段落，统一到英文或双语形式。
*   **验证**: 随机抽样对比微调模型与基线模型在风险合规提示语的准确度差异。

---

### 优先行动建议 (Action Plan)

1.  **扩展新闻数据集**: 运行 `python scripts/fetch_historical_news.py --start-date 2025-10-01 --end-date 2025-12-05 --tickers XAUUSD,GC=F --query "gold OR xauusd OR federal reserve"`，确认 `data/rag/news_archive.json` 覆盖所需窗口。
2.  **运行基准测试**: 使用 `backtest.py` 运行 Buy-and-Hold 和 SMA 策略，获取真实的基准数据。
3.  **生成交易日志**: 运行 Gold-Agent 在扩展数据集上的回测，并保存详细的 JSON 日志，以便在论文中引用具体的“高光时刻”交易案例。
4.  **整理微调语料**: 将最新的代理对话与风险审计记录导出为 JSONL，使用 `--dry-run` 选项验证格式后再触发实际微调任务。
