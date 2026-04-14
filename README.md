# PT2.3 - Text-to-SQL Agent

## 1. Team Members Identification

| Member | Full Name |
|--------|-----------|
| Member 1 | Serhii Turtsanash |
| Member 2 | Marti Serra |

### Work Distribution

- **Serhii Turtsanash**: Steps 1 and 2, evaluation system (`evals.py`) and test dataset.
- **Marti Serra**: Steps 3, 4 and 5 (CLI + Streamlit), general integration.

---

## Implementation Documentation

### Description

Text-to-SQL agent system that converts natural language questions into SQL queries on the Chinook database (SQLite), using GPT-4o-mini with LangGraph and LangChain.

| Step | Description |
|------|-------------|
| 1 | Basic agent without memory or HITL |
| 2 | Memory (checkpointing) + HITL on all tools |
| 3 | Selective HITL: only interrupts on risky operations |
| 4 | Automatic summarization of long conversations |
| 5 | Web interface with Streamlit |


### Design Decisions

**GPT-4o-mini**: Good balance of cost/speed/accuracy (100% on evaluations).

### Difficulties Encountered

**HITL in LangGraph**: Initially it interrupted on all tools, including safe ones. This was resolved with `es_herramienta_segura()` which auto-approves read operations.


**Memory in long conversations**: Without control, the history exceeded the model's context. `resumir_mensajes()` solves this by condensing old messages.

---

## 3. References

- **LangChain** - https://python.langchain.com/docs/
- **LangGraph** - https://langchain-ai.github.io/langgraph/
- **LangChain SQL Agent Tutorial** - https://python.langchain.com/docs/tutorials/sql_qa/
- **LangGraph HITL** - https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/
- **OpenAI API** - https://platform.openai.com/docs/
- **Streamlit** - https://docs.streamlit.io/
- **Chinook Database** - https://github.com/lerocha/chinook-database
- **SQLAlchemy** - https://docs.sqlalchemy.org/
