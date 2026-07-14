# CRAG-MM Agent

多模态检索增强生成（RAG）Agent，基于 **BaseAgent 框架** 实现，针对 KDD CUP 2025 CRAG-MM（Comprehensive RAG Benchmark - Multi Modal）赛道。

## 项目背景

2025 夏季学期智能技术实训课程项目，三人团队合作。在官方 Starter Kit 基础上，从 66.35% 幻觉率出发，通过五阶段优化将性能提升至可用水平。

## 核心优化

| 阶段 | 方法 | 效果 |
|------|------|------|
| 动态感知范式 | 将固定图像摘要重构成问题驱动的动态描述 | 提升视觉-语义对齐 |
| 二级检索管线 | BGE-reranker-large 对初筛结果重排序（Top10 → Top3） | 检索精度提升 |
| 自修正机制 | "草稿生成 → 事实核查"两阶段生成，幻觉防火墙 | 幻觉率 66.35% → 15.13% |
| 查询路由 | LLM 自规划：分析 query 决定是否需要 web 搜索 | 减少无效检索 |
| 云端推理 | Llama-3.2-11B-Vision + vLLM 推理框架 + GPU 云平台 | 全流程可复现 |

## 最终结果

- **幻觉率**：66.35% → **15.13%**（↓77%）
- **真实性分数**：-0.3462 → **0.0132**

## 技术栈

`vLLM` `Llama-3.2-11B-Vision` `BGE-reranker-large` `PyTorch` `Sentence-Transformers`

## 项目结构

```
.
├── my_rag_agent.py              # 核心 Agent（继承 BaseAgent，五阶段优化）
├── agents/
│   ├── base_agent.py            # CRAG-MM 官方 Agent 基类
│   ├── rag_agent.py             # 官方 SimpleRAGAgent 参考实现
│   ├── random_agent.py          # 随机 Agent 基线
│   ├── vanilla_llama_vision_agent.py  # Llama Vision Agent
│   └── user_config.py           # Agent 注册配置
├── docs/                        # 赛题文档
│   ├── dataset.md
│   ├── baselines.md
│   ├── submission.md
│   └── search_api.md
├── crag_batch_iterator.py       # 批处理迭代器
├── crag_image_loader.py         # 图像加载器
├── crag_web_result_fetcher.py   # Web 搜索结果获取
├── local_evaluation.py          # 本地评估脚本
├── utils.py                     # 工具函数
├── Dockerfile                   # Docker 镜像配置
├── docker_run.sh                # Docker 启动脚本
├── requirements.txt             # 依赖列表
└── aicrowd.json                 # AIcrowd 竞赛配置
```

## 快速开始

```bash
pip install -r requirements.txt
python local_evaluation.py
```

## 团队分工

三人合作项目，本仓库聚焦于 Agent 策略优化部分。完整项目使用算力云平台（AutoDL）租借 GPU 服务器运行。

## 参考

- 赛题：[KDD CUP 2025 CRAG-MM Challenge](https://www.aicrowd.com/challenges/meta-comprehensive-rag-benchmark-kdd-cup-2025)
- 框架：[meta-comprehensive-rag-benchmark-starter-kit](https://github.com/facebookresearch/meta-comprehensive-rag-benchmark-starter-kit)