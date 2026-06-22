# 电商多智能体协作系统 (MAS)

基于 **LangChain + LangGraph** 的电商多智能体协作系统，覆盖"商品咨询→下单引导→售后处理"的完整业务闭环。

## 🏗️ 系统架构

```
                        ┌─────────────────┐
                        │   路由Agent      │
                        │  (Router/Coord)  │
                        └──┬──────┬──────┬┘
                           │      │      │
              ┌────────────┘      │      └────────────┐
              ▼                   ▼                   ▼
      ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
      │  导购Agent    │  │  订单Agent    │  │  售后Agent    │
      │ (Shopping     │  │ (Order       │  │ (Aftersales  │
      │  Guide)       │  │  Agent)      │  │  Agent)      │
      └──────────────┘  └──────────────┘  └──────────────┘
            │                   │                   │
            └───────────────────┼───────────────────┘
                                │ (所有Agent通过Router协调)
                                ▼
                         LangGraph StateGraph
                    (状态管理 + 上下文传递 + 冲突仲裁)
```

### 五个核心状态

| 状态 | 说明 | 处理Agent |
|------|------|----------|
| **IDLE** | 就绪/等待输入 | Router |
| **INQUIRY** | 商品咨询 | Shopping Guide |
| **ORDER** | 订单处理 | Order Agent |
| **AFTERSALES** | 售后服务 | Aftersales Agent |
| **CROSS_AGENT** | 跨Agent协作 | Router调度 |

### 跨Agent协作流程示例

```
用户: "我买的耳机有质量问题，想退货然后换个同品牌更贵的"
  → Router: 意图=exchange_request, 路由到售后Agent
  → 售后Agent: 审核换货资格, 设置 task_chain=["product_inquiry", "new_order"]
  → Router: 弹出 product_inquiry, 路由到导购Agent
  → 导购Agent: 读取上下文(原订单信息), 推荐更高端产品
  → Router: 弹出 new_order, 路由到订单Agent
  → 订单Agent: 读取上下文(推荐商品), 创建新订单
  → Router: task_chain为空, 状态→IDLE, 结束
```

## 🚀 快速启动

### 前置条件

- Python 3.10+
- DeepSeek API Key ([获取地址](https://platform.deepseek.com))

### 安装

```bash
# 1. 进入项目目录
cd ex_project

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置环境变量
cp .env.example .env
# 编辑 .env 文件，填入你的 DEEPSEEK_API_KEY

# 4. 启动服务
python app.py
```

访问 **http://localhost:8000** 打开聊天界面。

> 首次启动会自动下载中文嵌入模型 (`BAAI/bge-small-zh-v1.5`, ~100MB) 并初始化ChromaDB向量数据库。

### 配置说明 (.env)

```bash
DEEPSEEK_API_KEY=sk-xxx        # 必填：DeepSeek API密钥
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
DEEPSEEK_MODEL=deepseek-chat
EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5
HOST=0.0.0.0
PORT=8000
```

## 📂 项目结构

```
ex_project/
├── app.py                  # FastAPI 应用入口
├── config.py               # 集中化配置管理
├── requirements.txt        # Python 依赖
├── .env.example            # 环境变量模板
├── README.md
│
├── src/
│   ├── agents/             # Agent 实现
│   │   ├── base_agent.py       # 抽象基类 (RAG Chain 模式)
│   │   ├── router_agent.py     # 路由Agent (意图识别 + 分发)
│   │   ├── shopping_guide.py   # 导购Agent (推荐 + 规格)
│   │   ├── order_agent.py      # 订单Agent (下单 + 物流)
│   │   └── aftersales_agent.py # 售后Agent (退换货 + 纠纷)
│   │
│   ├── state/              # LangGraph 状态管理
│   │   ├── schema.py           # AgentState TypedDict
│   │   ├── graph.py            # StateGraph 构建
│   │   └── transitions.py      # 意图→状态映射表
│   │
│   ├── communication/      # Agent间通信
│   │   ├── protocol.py         # JSON 消息协议
│   │   └── context_manager.py  # 上下文序列化
│   │
│   ├── rag/                # RAG 知识库
│   │   ├── loader.py           # JSON → Document
│   │   ├── store.py            # ChromaDB 管理
│   │   ├── retriever.py        # 领域检索
│   │   └── data/               # 示例数据
│   │       ├── products.json   # 5个商品
│   │       ├── policies.json   # 售后政策
│   │       └── faq.json        # 常见问答
│   │
│   └── conflict/           # 冲突解决
│       └── resolver.py         # 规则仲裁引擎
│
├── static/                 # 前端
│   ├── index.html              # 聊天界面 (SPA)
│   ├── style.css               # 样式 (暗色主题)
│   └── app.js                  # 前端逻辑 (SSE流式)
│
└── chroma_db/              # 向量数据库 (自动创建)
```

## 🔌 API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/` | 聊天界面 |
| `POST` | `/chat` | 发送消息 (JSON响应) |
| `POST` | `/chat/stream` | 发送消息 (SSE流式) |
| `GET` | `/session/{id}` | 查询会话状态 |
| `DELETE` | `/session/{id}` | 清除会话 |
| `GET` | `/health` | 健康检查 |
| `GET` | `/products` | 商品列表 |

### POST /chat 示例

```json
// Request
{ "message": "推荐一款蓝牙耳机", "session_id": "sess-abc" }

// Response
{
  "session_id": "sess-abc",
  "response": "根据您的需求，为您推荐...",
  "current_state": "INQUIRY",
  "intent": "product_recommend",
  "agent_trace": ["INQUIRY"]
}
```

## 🧪 验证清单

1. ✅ 单Agent查询：`推荐一款降噪耳机` → 导购Agent响应
2. ✅ 单Agent查询：`如何查询我的订单` → 订单Agent响应
3. ✅ 单Agent查询：`我要退货` → 售后Agent响应
4. ✅ 跨Agent协作：`我买的耳机有问题，想退货换一个更贵的` → 售后→导购→订单
5. ✅ 状态面板实时更新（IDLE→INQUIRY→ORDER→AFTERSALES→CROSS_AGENT）
6. ✅ RAG检索返回相关商品/政策信息

## 🛠️ 技术栈

- **Agent框架**: LangChain 0.3+
- **流程编排**: LangGraph 0.2+ (StateGraph + Command API)
- **LLM**: DeepSeek v4 Flash (OpenAI兼容API)
- **嵌入模型**: BAAI/bge-small-zh-v1.5 (sentence-transformers)
- **向量库**: ChromaDB (持久化)
- **后端**: FastAPI + Uvicorn
- **前端**: 原生 HTML/CSS/JS (SSE流式通信)
