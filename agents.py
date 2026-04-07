"""
Agente SQL - Pasos 1 a 4.
"""

from dotenv import load_dotenv
load_dotenv()

import os
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import HumanMessage, SystemMessage, RemoveMessage

# Base de datos
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Chinook.db").replace("\\", "/")
db = SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")

SYSTEM_PROMPT = """Eres un agente que interactua con una base de datos SQL.
Dada una pregunta, crea una consulta SQLite correcta, ejecutala y devuelve la respuesta.
Limita los resultados a 5 a menos que el usuario pida mas.
NO hagas operaciones DML (INSERT, UPDATE, DELETE, DROP).
Primero mira las tablas disponibles, luego consulta el esquema de las tablas relevantes."""

SYSTEM_PROMPT_V3 = """Eres un agente que interactua con una base de datos SQL.
Dada una pregunta, crea una consulta SQLite correcta, ejecutala y devuelve la respuesta.
Limita los resultados a 5 a menos que el usuario pida mas.
Primero mira las tablas disponibles, luego consulta el esquema de las tablas relevantes.

HERRAMIENTAS:
- Usa 'safe_sql_query' para consultas SELECT (lectura).
- Usa 'sql_db_query' SOLO para operaciones de escritura (INSERT, UPDATE, DELETE, DROP).
  Estas requieren aprobacion humana."""


# --- Paso 1 ---
def crear_agente(modelo="gpt-4o-mini"):
    """Agente simple sin memoria ni human-in-the-loop."""
    llm = ChatOpenAI(model=modelo)
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    agente = create_react_agent(
        model=llm,
        tools=toolkit.get_tools(),
        prompt=SYSTEM_PROMPT,
    )
    return agente


# --- Paso 2 ---
def crear_agente_v2(modelo="gpt-4o-mini"):
    """Agente con memoria y human-in-the-loop en todas las herramientas."""
    llm = ChatOpenAI(model=modelo)
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    memoria = InMemorySaver()
    agente = create_react_agent(
        model=llm,
        tools=toolkit.get_tools(),
        prompt=SYSTEM_PROMPT,
        checkpointer=memoria,
        interrupt_before=["tools"],
    )
    return agente


# --- Paso 3 ---
def crear_agente_v3(modelo="gpt-4o-mini"):
    """Human-in-the-loop solo para operaciones de riesgo."""
    llm = ChatOpenAI(model=modelo)
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    tools = toolkit.get_tools()

    # Herramienta segura clonada para SELECT
    safe_sql_tool = QuerySQLDataBaseTool(
        db=db,
        name="safe_sql_query",
        description=(
            "Usar SOLO para consultas SELECT/lectura. "
            "Entrada: sentencia SELECT unicamente. "
            "Sin DELETE/DROP/UPDATE."
        ),
    )

    # Mantener todas las herramientas excepto sql_db_query, anadir safe + sql_db_query
    sql_db_query_tool = next(t for t in tools if t.name == "sql_db_query")
    tools_v3 = [t for t in tools if t.name != "sql_db_query"] + [safe_sql_tool, sql_db_query_tool]

    memoria = InMemorySaver()
    agente = create_react_agent(
        model=llm,
        tools=tools_v3,
        prompt=SYSTEM_PROMPT_V3,
        checkpointer=memoria,
        interrupt_before=["tools"],
    )
    return agente


# --- Paso 4 ---
def crear_agente_v4(modelo="gpt-4o-mini", max_mensajes=20):
    """Como v3 pero con gestion de memoria (resumen)."""
    llm = ChatOpenAI(model=modelo)
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    tools = toolkit.get_tools()

    safe_sql_tool = QuerySQLDataBaseTool(
        db=db,
        name="safe_sql_query",
        description=(
            "Usar SOLO para consultas SELECT/lectura. "
            "Entrada: sentencia SELECT unicamente. "
            "Sin DELETE/DROP/UPDATE."
        ),
    )

    sql_db_query_tool = next(t for t in tools if t.name == "sql_db_query")
    tools_v4 = [t for t in tools if t.name != "sql_db_query"] + [safe_sql_tool, sql_db_query_tool]

    memoria = InMemorySaver()
    agente = create_react_agent(
        model=llm,
        tools=tools_v4,
        prompt=SYSTEM_PROMPT_V3,
        checkpointer=memoria,
        interrupt_before=["tools"],
    )
    return agente, llm, max_mensajes


# --- Utilidades ---
HERRAMIENTAS_SEGURAS = {"safe_sql_query", "sql_db_list_tables", "sql_db_schema", "sql_db_query_checker"}


def es_herramienta_segura(tool_calls):
    """True si todas las herramientas pendientes son seguras."""
    return all(tc["name"] in HERRAMIENTAS_SEGURAS for tc in tool_calls)


def resumir_mensajes(llm, mensajes, max_mensajes=20):
    """Resumir mensajes antiguos si superan el limite.
    Usa RemoveMessage para borrar los viejos y anade un SystemMessage con el resumen."""
    if len(mensajes) <= max_mensajes:
        return None  # No hace falta resumir

    a_resumir = mensajes[:-6]
    recientes = mensajes[-6:]

    texto = "\n".join(
        f"{m.type}: {m.content}"
        for m in a_resumir
        if hasattr(m, "content") and m.content
    )

    resumen = llm.invoke([
        HumanMessage(content=f"Resume brevemente esta conversacion:\n{texto}")
    ]).content

    # Borrar mensajes viejos + anadir resumen
    borrar = [RemoveMessage(id=m.id) for m in a_resumir if hasattr(m, "id")]
    nuevo = SystemMessage(content=f"Resumen de la conversacion anterior: {resumen}")
    return borrar + [nuevo] + recientes


def preguntar(agente, pregunta):
    """Envia una pregunta al agente y devuelve la respuesta (Paso 1, sin memoria)."""
    resultado = agente.invoke({"messages": [{"role": "user", "content": pregunta}]})
    return resultado["messages"][-1].content