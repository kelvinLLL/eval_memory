# Agentic Memory 评测框架设计方案（v4）

---

## 一、设计原则

1. **轻量第一**：纯 Python、< 1500 行、无 Docker/K8s/沙盒依赖
2. **复用数据集**：不造新 benchmark，适配已有的
3. **可信指标**：每个指标必须有 ≥ 3 个业界项目背书
4. **统一适配**：一个 Dataset Adapter 层屏蔽所有 benchmark 差异；一个 Memory System 接口（3 个方法）屏蔽所有被评系统差异

---

## 二、各 Benchmark 接口差异分析

> **为什么要先做这件事**：我们要复用 LoCoMo、LongMemEval 等现有数据集，但它们的数据格式、注入协议、评分方式各不相同。只有逐一拆解后，才能提取出最小公共抽象，设计出一个真正通用的 adapter 层。

### 2.1 LoCoMo（Snap Research, ACL 2024）

| 维度 | 详情 |
|------|------|
| **数据格式** | 单个 JSON（`locomo10.json`），10 条 conversation |
| **数据结构** | `{speaker_a, speaker_b, session_1..N, session_N_date_time, session_N_summary, qa: [{question, answer, category, evidence}]}` |
| **注入粒度** | **session 级**：每个 session 包含多个 turns，整体作为一个注入单元 |
| **注入方式** | 被动、顺序注入所有 sessions |
| **时间信息** | ✅ 每个 session 有 `date_time` 字段 |
| **评测时机** | 全部 sessions 注入后，统一提问 |
| **答案格式** | 自由文本 |
| **评分** | LLM-as-Judge 二值 (CORRECT/WRONG) |
| **评分排除** | Category 5 (adversarial) 不计入最终分数 |
| **使用者** | Mem0, Zep/Graphiti, MemMachine, Backboard, Letta, GAM |

**已知坑**：Category ID 与论文描述不一致（需以代码为准）；Mem0 vs Zep 在 LoCoMo 上的评分争议 [github.com/getzep/zep-papers/issues/5] 证明了评测协议标准化的重要性。

### 2.2 LongMemEval（ICLR 2025）

| 维度 | 详情 |
|------|------|
| **数据格式** | JSON（`longmemeval_s.json` ~115K tokens / `longmemeval_m.json` ~1.5M tokens / `longmemeval_oracle.json` 含 ground truth） |
| **数据结构** | 4-tuple `(S, q, t_q, a)`：S = 按时序排列的 sessions（每个含多轮 user-assistant 对话），q = 问题，t_q = 问题时间戳，a = 答案短语或 rubric |
| **注入粒度** | 可选 **turn 级** 或 **session 级**（论文推荐 turn 级，Supermemory 用 session 级） |
| **注入方式** | 被动、顺序注入 |
| **时间信息** | ✅ 每个 turn 和 session 均有时间戳 |
| **评测时机** | 全部 history 注入后，在新 session 中提问 |
| **答案格式** | 短语 或 自然语言 rubric |
| **评分** | GPT-4o judge（与人类 >97% 一致性）|
| **特殊评分** | 若系统暴露检索结果 → 可额外计算 Recall@K, NDCG@K |
| **输出格式** | JSONL，每行 `{question_id, hypothesis}` |
| **5种能力标签** | IE (信息抽取) / MR (多会话推理) / TR (时间推理) / KU (知识更新) / ABS (拒绝回答) |
| **使用者** | Supermemory, Mem0, LiCoMemory, ENGRAM-R, GAM, MemAgent |

### 2.3 MemoryAgentBench / MemAE（ICLR 2026）

| 维度 | 详情 |
|------|------|
| **数据格式** | HuggingFace 数据集（自动下载），含多个子集 |
| **数据结构** | 每个子集 = `{chunks[], questions[]}`，chunk 为 512 或 4096 tokens 的文本块 |
| **子集分类** | AR (Accurate Retrieval, 改造自 RULER/InfBench) / TTL (Test-time Learning) / LRU (Long-range Understanding, 改造自 LongMemEval_S*) / CR (Conflict Resolution, 自建 FactConsolidation) |
| **注入粒度** | **chunk 级**：增量注入 512/4096 token 的 chunks |
| **注入方式** | 被动、增量逐块喂入 |
| **时间信息** | ❌ 无时间戳 |
| **评测时机** | 全部 chunks 注入后统一提问 |
| **答案格式** | 精确答案 (AR/CR) 或 自由文本 (LRU) |
| **评分** | Exact Match (AR) + LLM-as-Judge (LRU, longmemeval/infbench_sum) |
| **代码结构** | `main.py` → `agent.py`（定义各类 agent）→ `methods/`（RAG/Mem0/Letta/Cognee 实现） |
| **Agent 接口** | 每类 agent 通过 config 选择，执行 `ingest_chunks → answer_questions` |
| **使用者** | 论文内评估了 Mem0, Letta, Cognee, HippoRAG, long-context LLMs, RAG baselines |

### 2.4 接口差异总结 → 提取公共抽象

```
                LoCoMo          LongMemEval     MemAE
               ─────────       ─────────────   ──────
注入单元:       session          turn/session    chunk
时间信息:       ✅ session级     ✅ turn级        ❌
评测时机:       注入后统一问     注入后统一问     注入后统一问  ← 一致！
答案格式:       自由文本         短语/rubric     精确/自由     ← 都是文本
评分方式:       LLM judge       LLM judge       EM + LLM     ← 2种覆盖

最大公约数:
  注入 = 一段文本 + 可选元数据（时间戳等）
  评测 = 一个问题 + 一个期望答案 + 一个评分方式
  流程 = 顺序注入 → 完成后提问 → 评分
```

**关键洞察**：三个 benchmark 的评测流程高度一致（顺序注入 → 统一提问 → 评分），差异仅在于注入粒度和评分方式。这意味着我们可以用一个极简的公共抽象覆盖所有三个 benchmark。

---

## 三、统一 Adapter 层

### 3.1 数据结构（最小抽象）

```python
@dataclass
class MemoryChunk:
    """注入单元 —— 从上面的分析中提取的最小公共结构"""
    content: str                        # 必须：文本内容
    timestamp: Optional[str] = None     # 可选：ISO 8601（LoCoMo session级, LongMemEval turn级）
    metadata: dict = field(default_factory=dict)  # 可选：speaker, session_id 等

@dataclass
class EvalQuestion:
    """评测问题"""
    question_id: str
    question: str
    expected_answer: str
    category: str                       # 能力标签（统一映射后的）
    eval_method: str = "llm_judge"      # "llm_judge" | "exact_match"
    timestamp: Optional[str] = None     # 问题时间戳（LongMemEval 有）

@dataclass
class EvalCase:
    """一条评测用例 = 一段注入序列 + 一组问题"""
    case_id: str
    chunks: List[MemoryChunk]
    questions: List[EvalQuestion]
    source: str                         # "locomo" | "longmemeval" | "memae" | "netops"
```

### 3.2 Dataset Adapter 接口

```python
class DatasetAdapter(ABC):
    """每个 benchmark 实现一个：原始数据 → List[EvalCase]"""

    @abstractmethod
    def load(self, path: str) -> List[EvalCase]:
        """加载并转换"""

    def get_config(self) -> dict:
        """该 benchmark 的默认评测配置"""
        return {"eval_method": "llm_judge", "exclude_categories": []}
```

### 3.3 各 Adapter 的转换逻辑

**LoCoMoAdapter**：
```
每个 conversation → 1 个 EvalCase
  session_1..N → N 个 MemoryChunk
    content = 所有 turns 拼接（"{speaker}: {text}\n..."）
    timestamp = session_N_date_time
  qa 中 category ∈ {1,2,3,4} → EvalQuestion
    category 映射: 1→single_hop, 2→multi_hop, 3→temporal, 4→open_domain
    eval_method = "llm_judge"
```

**LongMemEvalAdapter**：
```
整个数据集 → 多个 EvalCase（按 question 分组，共享 sessions）
  sessions → MemoryChunk 序列
    粒度可配: session 级（拼接所有 turns）或 turn 级（每个 turn 一条）
    timestamp = session/turn 的时间戳
  每个 question → EvalQuestion
    category = IE/MR/TR/KU/ABS
    eval_method = "llm_judge"
```

**MemAEAdapter**（可选，Phase 2+）：
```
HuggingFace 数据集 → 多个 EvalCase
  chunks → MemoryChunk 序列（无 timestamp）
  questions → EvalQuestion
    category = AR/TTL/LRU/CR
    eval_method = AR→"exact_match", 其余→"llm_judge"
```

### 3.4 Memory System 接口（3 个方法）

```python
class MemorySystem(ABC):
    """被评系统只需实现 3 个方法"""

    @abstractmethod
    def reset(self):
        """清空状态，为新一轮评测做准备"""

    @abstractmethod
    def ingest(self, content: str, metadata: dict = {}) -> dict:
        """注入一条 memory
        Returns: {"latency_ms": float, "tokens_used": int}
        """

    @abstractmethod
    def answer(self, query: str, query_metadata: dict = {}) -> dict:
        """回答问题
        Returns: {"answer": str, "latency_ms": float, "tokens_used": int}
        """
```

> **设计决策**：只有 3 个方法，不要 `retrieve()`、`update()`、`delete()` 等。原因是：不同 memory 系统的检索/更新逻辑差异极大（有些在 ingest 时自动更新，有些需要显式调用），强行统一这些接口会导致适配成本远高于收益。我们只关心最终效果——你 ingest 了什么、answer 了什么。

### 3.5 评测执行流程

```python
# runner.py 核心逻辑（伪代码，< 100 行）

for case in dataset_adapter.load(path):
    system.reset()

    # 注入阶段
    for chunk in case.chunks:
        stats = system.ingest(chunk.content, chunk.metadata)
        record(stats)

    # 评测阶段
    for q in case.questions:
        result = system.answer(q.question, {"timestamp": q.timestamp})
        score = grade(q, result["answer"])  # exact_match 或 llm_judge
        results.append({
            "case_id": case.case_id,
            "question_id": q.question_id,
            "category": q.category,
            "score": score,
            "latency_ms": result["latency_ms"],
            "tokens_used": result["tokens_used"],
        })

# 聚合 + 报告
aggregate_by_category(results)
generate_markdown_report(results)
```

---

## 四、被评系统分类与清单

### 4.1 三类记忆架构 + 上界 baseline

| 类别 | 架构特征 | 代表系统 | Phase 1 适配 |
|------|----------|----------|-------------|
| **Long-context LLM** *(上界)* | 全量 chunks 塞进 prompt，无记忆系统 | GPT-4.1 / Claude | ✅ |
| **RAG + LLM** | Embedding 检索 + LLM 生成，无自主记忆管理 | MemU, 自建 SimpleRAG | ✅ |
| **Agentic Memory** | Agent 自主决定存/取/更新/删除，有记忆生命周期管理 | Mem0, MemGPT/Letta | ✅ |
| **Embedding-only** *(下界)* | 纯 embedding 相似度检索 + 拼接 context | FAISS/Chroma top-k → LLM | ✅ |

### 4.2 具体实现

```python
# systems/ 目录
systems/
├── base.py               # MemorySystem ABC
├── long_context.py        # 上界：全部 chunks 拼接进 prompt
├── embedding_only.py      # 下界：FAISS top-k → 拼接 context → LLM
├── simple_rag.py          # RAG+LLM（MemU 类架构）
├── mem0_system.py         # Agentic: Mem0 API wrapper
└── letta_system.py        # Agentic: Letta/MemGPT (可选)
```

| 系统 | ingest 实现 | answer 实现 | 特殊处理 |
|------|------------|------------|---------|
| Long-context | 累积 chunk 到内部 buffer | 全部 buffer + question → LLM | token 上限截断 |
| Embedding-only | chunk → embedding → FAISS | query embedding → top-k → 拼接 → LLM | 无重排、无 chunking |
| SimpleRAG (MemU类) | chunk → split → embed → vector store | query → 检索 + rerank → LLM 生成 | 可配 chunk_size, top_k |
| Mem0 | `mem0.add(content, user_id)` | `mem0.search(query) → context → LLM` | 需 API key |
| Letta | `letta.send_message(content)` | `letta.send_message(query)` | 需运行 Letta server |

---

## 五、评测指标（每个指标 ≥ 3 个业界引用）

### 5.1 核心指标（必选，MVP 就需要）

| 指标 | 定义 | 业界使用方 (≥3) | 引用 |
|------|------|-----------------|------|
| **Answer Accuracy (LLM-judge)** | GPT-4o 对 (question, expected, predicted) 做二值判定 CORRECT/WRONG | LoCoMo, LongMemEval, Mem0, Zep/Graphiti, MemMachine, Backboard, Supermemory, Letta, MEMTRACK, GAM | [1][2][5][6][10][12] |
| **Category-wise Accuracy** | 按能力类别分别统计 accuracy | LoCoMo (4类), LongMemEval (5类), MemAE (4类), Supermemory, Mem0 | [1][2][3][10][5] |
| **Token Cost** | 每次 ingest/answer 的 token 消耗总量 | MEMTRACK, ENGRAM-R (降低95.5%), AIOpsLab, Letta (cost-aware) | [4][6][9] |

### 5.2 扩展指标（Phase 2+ 按需开启）

| 指标 | 定义 | 业界使用方 (≥3) | 引用 |
|------|------|-----------------|------|
| **Retrieval Latency P95** | 检索操作第 95 百分位延迟 | Mem0, Supermemory, LiCoMemory, MemMachine | [5][10] |
| **Conflict Resolution Accuracy** | 矛盾信息更新后是否给出最新答案 | MemAE (CR子集), MEMTRACK, Mem0^g | [3][4][5] |
| **Abstention Accuracy** | 该拒绝时是否正确拒绝 | LongMemEval (30 ABS questions), Letta | [2][6] |
| **Multi-run 标准差** | 多次运行结果的方差 | Mem0 论文 (10次均值+std), Backboard | [5][12] |

### 5.3 不选的指标（避免过度设计）

- Recall@K / NDCG@K：仅当系统暴露检索中间结果时有意义，不通用
- ROUGE / BLEU / FactScore：用于摘要任务，与 memory QA 评测场景不匹配
- Tool Call Efficiency：需要沙盒环境支持，太重
- Step Efficiency：需要定义"步骤"，运维场景尚未标准化

---

## 六、评测层级与数据集

### Layer 1：通用 Memory 能力（Phase 1 两个都跑）

| 数据集 | 覆盖能力 | 规模 | 选择理由 |
|--------|----------|------|----------|
| **LongMemEval_S** | IE + MR + TR + KU + ABS | 500 questions, ~115K tokens | 社区标准，Supermemory/Mem0/LiCoMemory/ENGRAM-R/GAM/MemAgent 均使用 |
| **LoCoMo** | single-hop + multi-hop + temporal + open-domain | ~1986 questions, 10 conversations | 另一社区标准，Mem0/Zep/MemMachine/Backboard/GAM 均使用 |

两者互补：LongMemEval 独有知识更新 (KU) 和拒绝 (ABS)；LoCoMo 独有 multi-hop 推理和更大的 QA 规模。

**不选**：MemAE (pipeline 重)、Letta Leaderboard (绑框架)、MEMTRACK (需沙盒)。

### Layer 2：网络运维专用 —— AIOps 公开数据集方案

> **背景**：暂无内部运维数据，从 **告警** 和 **排障** 两个方向切入，利用 AIOps 社区已有的竞赛数据集和公开 benchmark 构建评测场景。

#### 6.1 可用的 AIOps 公开数据源

| 数据集 | 来源 | 数据类型 | 获取方式 | 适用场景 |
|--------|------|----------|----------|----------|
| **OpsEval** | 清华 NetMan + 11家企业, FSE'25 | 7184 MC + 1736 QA，中英双语，覆盖8种运维任务 | GitHub: `NetManAIOps/OpsEval-Datasets`; HuggingFace: `Junetheriver/OpsEval` (公开20%) | Runbook/SOP 知识记忆 + QA |
| **AIOps Challenge 2020** | 清华 NetMan 竞赛 | 业务指标 + 平台指标 + 调用链 (traces) | GitHub: `NetManAIOps/AIOps-Challenge-2020-Data` | 故障诊断时序记忆 |
| **AIOps Challenge 2021** | 清华 NetMan 竞赛 | Logs + Metrics + Traces，多模态异常检测 | 竞赛平台 (aiops.cn) | 多模态告警关联 |
| **DejaVu Dataset** | 清华 NetMan, FSE'22 | 微服务故障诊断（metrics + failure labels + FDG） | Zenodo: `10.5281/zenodo.6955909` | 重复故障模式记忆 |
| **SOFI Dataset** | ScienceDirect'22 | IP网络 症状-故障 因果关系 | 论文附带数据集 | 网络故障因果知识 |
| **5G RCA Challenge** | SRIBD AIOps 竞赛 | 真实 5G 网络故障定位 + 因果图 | `aiops.sribd.cn` | 无线网络故障记忆 |
| **ITBench** | IBM, ICML'25 | 102个 SRE/CISO/FinOps 真实场景 | GitHub: `itbench-hub/ITBench` | Agent 排障场景参考 |
| **AIOpsLab** | Microsoft, MLSys'25 | 微服务故障注入 + 遥测数据 | GitHub: `microsoft/AIOpsLab` | 端到端排障流程参考 |

#### 6.2 转换策略：从 AIOps 数据到 Memory 评测

**核心思路**：AIOps 数据集大多是结构化的（时序/Trace/日志），不能直接当 memory QA 用。需要做 **文本化转换 + QA 构造**，将其变成 `EvalCase` 格式。

**场景 A：告警知识记忆（基于 OpsEval）**

```
数据源: OpsEval QA 数据集（已有 question + answer）
转换:
  1. 从 OpsEval 中筛选 "告警处理"、"故障诊断"、"根因分析" 子领域
  2. 将 QA pairs 拆分为:
     - 知识注入: 把 answer 改写为 SOP/Runbook 片段 → MemoryChunk
     - 测试问题: 用原始 question（或改写变体）→ EvalQuestion
  3. 模拟时序: 按 topic 分组注入，模拟"逐步积累运维知识"

示例 EvalCase:
  chunks = [
    "BGP邻居状态机包含6个状态: Idle, Connect, Active, OpenSent, 
     OpenConfirm, Established...",
    "当BGP邻居频繁flap时，首先检查物理链路是否抖动，然后检查
     hold timer配置...",
    "BGP路由黑洞常见原因: 1)下一跳不可达 2)策略过滤 3)聚合路由
     配置错误..."
  ]
  questions = [
    {q: "BGP邻居反复在Active和Connect之间震荡，可能的原因？",
     a: "TCP连接建立失败...", category: "IE"}
  ]
```

**场景 B：故障模式记忆（基于 DejaVu + AIOps Challenge）**

```
数据源: DejaVu fault types + AIOps Challenge incident 数据
转换:
  1. 故障事件 → 自然语言叙述:
     "2020-03-15 14:23 告警: ts-order-service 响应时间从 120ms 飙升至
      2300ms。关联告警: ts-travel-service CPU 95%。根因: 内存泄漏导致
      GC 频繁。处理: 重启 ts-travel-service pod。"
  2. 多个历史故障案例 → MemoryChunk（带时间戳）
  3. 新故障场景作为 question → "ts-order-service 再次响应慢，
     有哪些类似的历史案例？"

评测能力:
  - 相似故障匹配   → IE / MR
  - 故障时间线理解 → TR
  - 处理方案更新   → KU
```

**场景 C：告警风暴去重与关联（半合成）**

```
参考: ICSE'20 "Understanding and Handling Alert Storm"
构造:
  1. 生成告警流:
     "[10:01] CRITICAL router-core-01 BGP peer 192.168.1.1 DOWN"
     "[10:01] WARNING  switch-agg-03 interface Gi0/1 flapping"
     "[10:02] CRITICAL router-core-01 OSPF neighbor 10.0.0.2 LOST"
     ... (10+ alerts, 部分重复/关联)
  2. 每条告警 → MemoryChunk（带精确时间戳）
  3. 评测问题:
     - "当前有多少条不重复的告警？" → 去重
     - "router-core-01 的故障根因最可能是什么？" → 关联推理
     - "之前遇到过 BGP+OSPF 同时 down 吗？" → 历史模式匹配
```

#### 6.3 NetOps Adapter 设计

```python
class NetOpsAdapter(DatasetAdapter):
    """统一处理三种运维场景，输入格式:
    {
      "case_id": str,
      "chunks": [{"content": str, "timestamp": str?, "metadata": {}}],
      "questions": [{"question": str, "answer": str, "category": str}],
      "source": "netops"
    }
    """
    def load(self, path: str) -> List[EvalCase]: ...
```

#### 6.4 能力标签映射

```
通用 Memory 能力        →  运维场景对应
───────────────────────────────────────────
IE (信息抽取)           →  从 SOP 中提取具体步骤
MR (多会话推理)         →  跨多次故障案例推理根因
TR (时间推理)           →  告警时间线排序与因果
KU (知识更新)           →  SOP 更新后使用新版本
ABS (拒绝回答)          →  无相关历史案例时说"不知道"
CR (冲突解决, from MemAE) → 同一故障不同处理方案取最新
```

---

## 七、项目结构

```
memory-eval/
├── config.yaml                  # 唯一配置入口
├── adapters/                    # Dataset Adapter 层
│   ├── base.py                  # MemoryChunk, EvalQuestion, EvalCase, DatasetAdapter ABC
│   ├── locomo.py                # LoCoMoAdapter
│   ├── longmemeval.py           # LongMemEvalAdapter
│   └── netops.py                # NetOpsAdapter (Phase 2)
├── systems/                     # Memory System 接口层
│   ├── base.py                  # MemorySystem ABC (reset/ingest/answer)
│   ├── long_context.py          # 上界 baseline：全量塞 prompt
│   ├── embedding_only.py        # 下界 baseline：纯 FAISS top-k
│   ├── simple_rag.py            # RAG+LLM（MemU 类架构）
│   ├── mem0_system.py           # Agentic: Mem0
│   └── letta_system.py          # Agentic: Letta/MemGPT (可选)
├── graders/
│   ├── llm_judge.py             # GPT-4o judge (遵循 LongMemEval 协议)
│   └── exact_match.py           # 归一化精确匹配
├── runner.py                    # 评测主循环 (< 150 行)
├── report.py                    # 结果聚合 + Markdown 输出 (< 100 行)
├── scripts/                     # 数据转换脚本 (Phase 2)
│   ├── convert_opseval.py       # OpsEval → netops EvalCase
│   ├── convert_dejavu.py        # DejaVu → netops EvalCase
│   └── gen_alert_storm.py       # 半合成告警流生成
└── data/
    ├── locomo10.json
    └── longmemeval_s.json
```

目标：**总代码 < 1500 行**（不含 scripts/），一行命令跑通：
```bash
python runner.py --config config.yaml
```

---

## 八、配置

```yaml
seed: 42

datasets:
  - name: longmemeval_s
    adapter: longmemeval
    path: data/longmemeval_s.json
    granularity: session       # turn | session

  - name: locomo
    adapter: locomo
    path: data/locomo10.json
    exclude_adversarial: true  # 社区标准：排除 category 5

systems:
  # 上界 baseline
  - name: long_context_baseline
    type: long_context
    model: gpt-4.1

  # 下界 baseline (纯 embedding)
  - name: embedding_only
    type: embedding_only
    embedding: text-embedding-3-small
    top_k: 10

  # RAG+LLM (MemU 类)
  - name: simple_rag
    type: simple_rag
    embedding: text-embedding-3-small
    top_k: 5
    llm: gpt-4o-mini

  # Agentic Memory
  - name: mem0
    type: mem0
    api_key: ${MEM0_API_KEY}

grading:
  judge_model: gpt-4o
  num_runs: 3

output:
  format: markdown             # markdown | json
```

---

## 九、实施路线

| Phase | 时间 | 交付物 | 核心任务 |
|-------|------|--------|----------|
| **1: MVP** | 1 周 | 4类系统 × 2个数据集的完整对比 | `base.py` + 2 Adapters + 4 Systems + `runner.py` + `llm_judge.py` |
| **2: 深化** | 1-2 周 | 扩展指标 + 详细分析报告 | Token cost + multi-run std + category 细化 + Letta 适配 |
| **3: NetOps** | 2-3 周 | 运维专用评测 | OpsEval/DejaVu 数据转换 + 告警场景半合成 + `NetOpsAdapter` |

### Phase 1 详细任务分解

```
Week 1:
├── Day 1-2: 框架骨架
│   ├── base.py (数据结构 + ABC 接口)
│   ├── runner.py (主循环)
│   ├── graders/llm_judge.py + exact_match.py
│   └── report.py (Markdown 输出)
│
├── Day 2-3: 数据适配
│   ├── adapters/locomo.py  → 验证: 10 EvalCase, ~1986 questions
│   └── adapters/longmemeval.py → 验证: N EvalCase, 500 questions
│
├── Day 3-4: 系统适配
│   ├── systems/long_context.py      (全量 prompt)
│   ├── systems/embedding_only.py    (FAISS + LLM)
│   ├── systems/simple_rag.py        (chunk + embed + retrieve + LLM)
│   └── systems/mem0_system.py       (Mem0 API)
│
├── Day 5: 跑通 + 报告
│   ├── 4 systems × {LoCoMo, LongMemEval_S}
│   ├── Markdown 对比表
│   └── Sanity check: long_context ≥ others ≥ embedding_only
```

**Phase 1 预期输出示例**:

```markdown
## Memory System Evaluation Report

### LongMemEval_S (500 questions)

| System          | Overall | IE   | MR   | TR   | KU   | ABS  | Tokens | 
|-----------------|---------|------|------|------|------|------|--------|
| long_context    | 78.2%   | 85%  | 72%  | 80%  | 75%  | 70%  | 1.2M   |
| simple_rag      | 62.4%   | 70%  | 55%  | 58%  | 65%  | 60%  | 180K   |
| mem0            | 58.8%   | 68%  | 52%  | 55%  | 60%  | 55%  | 95K    |
| embedding_only  | 45.6%   | 55%  | 38%  | 40%  | 48%  | 42%  | 120K   |

### LoCoMo (10 conversations, ~1986 questions)
... (similar table)
```

---

## 十、参考文献

| # | 论文/项目 | 我们引用了什么 |
|---|----------|---------------|
| [1] | LoCoMo (Maharana+ ACL'24) | 数据集 + LLM judge 协议 + category 体系 |
| [2] | LongMemEval (Wu+ ICLR'25) | 数据集 + GPT-4o judge + 5 种能力分类 |
| [3] | MemoryAgentBench/MemAE (Hu+ ICLR'26) | 4 核心能力框架 (AR/TTL/LRU/CR) |
| [4] | MEMTRACK (NeurIPS SEA'25) | Efficiency + Redundancy 指标 |
| [5] | Mem0 (mem0.ai '25) | 多次运行评测规范 |
| [6] | Letta Leaderboard ('25) | SimpleQA 评分 + memory penalty |
| [7] | Evo-Memory (NeurIPS'25) | Sequence Robustness 概念 |
| [8] | OpsEval (清华NetMan, FSE'25) | 7184 MC + 1736 QA 运维知识 |
| [9] | AIOps Challenge 2020/2021 (清华NetMan) | 竞赛数据: 指标+调用链+日志 |
| [10] | Supermemory (supermemory.ai) | LongMemEval session 级注入 |
| [11] | Zep vs Mem0 争议 (GitHub) | 评测协议标准化教训 |
| [12] | Backboard LoCoMo Benchmark | LoCoMo 评测实现参考 |
| [13] | DejaVu (Li+ FSE'22) | 微服务故障诊断数据 (Zenodo) |
| [14] | ITBench (IBM, ICML'25) | 102 SRE 场景 |
| [15] | AIOpsLab (Microsoft, MLSys'25) | 微服务故障注入框架 |
| [16] | SOFI Dataset (ScienceDirect'22) | IP网络症状-故障因果 |
| [17] | 5G RCA Challenge (SRIBD) | 5G 网络故障定位数据 |
| [18] | Alert Storm (ICSE'20, 清华NetMan) | 告警风暴处理模式 |

---

## 附录：v3 → v4 变更说明

| 变更项 | v3 | v4 |
|--------|----|----|
| 被评系统 | 仅列 MemU + 2 baselines | 明确4类：long-context上界 / RAG+LLM / Agentic / Embedding下界 |
| Phase 1 数据集 | 未确定 | 两个都跑：LoCoMo + LongMemEval_S |
| Layer 2 数据源 | 仅提 NIKA/OpsEval，3行描述 | 8个公开数据源 + 3种转换场景 + 能力映射 |
| systems/ | 3个文件 | 6个文件，含上下界 baseline |
| 参考文献 | 12条 | 18条，新增 AIOps 竞赛/框架 |
| 预期输出 | 无 | 含 Phase 1 报告示例 |