"""
Agente SQL - Paso 1 y Paso 2
Paso 1: sin memoria, sin human-in-the-loop.
Paso 2: con memoria (InMemorySaver) y human-in-the-loop en todas las consultas.
"""

from dotenv import load_dotenv
load_dotenv()

import os
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver

# Conexion a la base de datos (ruta absoluta con barras /)
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Chinook.db").replace("\\", "/")
db = SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")

# System prompt del agente
SYSTEM_PROMPT = """Eres un agente que interactua con una base de datos SQL.
Dada una pregunta, crea una consulta SQLite correcta, ejecutala y devuelve la respuesta.
Limita los resultados a 5 a menos que el usuario pida mas.
NO hagas operaciones DML (INSERT, UPDATE, DELETE, DROP).
Primero mira las tablas disponibles, luego consulta el esquema de las tablas relevantes."""


def crear_agente(modelo="gpt-4o-mini"):
    """Paso 1: agente simple sin memoria ni human-in-the-loop."""
    llm = ChatOpenAI(model=modelo)
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    tools = toolkit.get_tools()

    agente = create_react_agent(
        model=llm,
        tools=tools,
        prompt=SYSTEM_PROMPT,
    )
    return agente


def crear_agente_v2(modelo="gpt-4o-mini"):
    """Paso 2: agente con memoria y human-in-the-loop en todas las herramientas."""
    llm = ChatOpenAI(model=modelo)
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    tools = toolkit.get_tools()

    memoria = InMemorySaver()

    agente = create_react_agent(
        model=llm,
        tools=tools,
        prompt=SYSTEM_PROMPT,
        checkpointer=memoria,
        interrupt_before=["tools"],
    )
    return agente


def preguntar(agente, pregunta):
    """Envia una pregunta al agente y devuelve la respuesta (Paso 1)."""
    resultado = agente.invoke({"messages": [{"role": "user", "content": pregunta}]})
    return resultado["messages"][-1].content
