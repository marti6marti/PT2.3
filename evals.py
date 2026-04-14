

import json
import time
from agents import crear_agente, preguntar, db, SYSTEM_PROMPT

# Dataset: 5 simples + 5 complejas
DATASET = {
    "simples": [
        ("Cuantos artistas hay?", "275"),
        ("Cuantos empleados hay?", "8"),
        ("Cuantos generos hay?", "25"),
        ("Cuantos albums hay?", "347"),
        ("Cual es el email de Nancy Edwards?", "nancy@chinookcorp.com"),
    ],
    "complejas": [
        ("Que artista tiene mas albums?", "Iron Maiden"),
        ("Cual es el pais con mas clientes?", "USA"),
        ("Cuantas pistas son del genero Rock?", "1297"),
        ("Que empleado tiene mas clientes asignados?", "Jane Peacock"),
        ("Cuantas pistas tiene el album 'Let There Be Rock'?", "8"),
    ],
}

DELAY = 2
MAX_REINTENTOS = 2


def preguntar_con_reintento(agente, pregunta):
    for intento in range(MAX_REINTENTOS + 1):
        try:
            return preguntar(agente, pregunta)
        except Exception as e:
            if intento < MAX_REINTENTOS:
                espera = 5 * (intento + 1)
                print(f"       Reintentando en {espera}s... ({e.__class__.__name__})")
                time.sleep(espera)
            else:
                raise


def evaluar(modelo="gpt-4o-mini", prompt=None):
    if prompt:
        from langchain_openai import ChatOpenAI
        from langchain_community.agent_toolkits import SQLDatabaseToolkit
        from langgraph.prebuilt import create_react_agent

        llm = ChatOpenAI(model=modelo, timeout=60, max_retries=2)
        toolkit = SQLDatabaseToolkit(db=db, llm=llm)
        agente = create_react_agent(model=llm, tools=toolkit.get_tools(), prompt=prompt)
    else:
        agente = crear_agente(modelo)

    print(f"\n{'='*50}")
    print(f"Modelo: {modelo}")
    print(f"Prompt: {'CONCISO' if prompt and prompt != SYSTEM_PROMPT else 'DEFAULT'}")
    print(f"{'='*50}")

    resultados = {}

    for tipo, pares in DATASET.items():
        aciertos = 0
        errores = 0
        print(f"\n--- {tipo.upper()} ---")

        for pregunta, esperado in pares:
            try:
                respuesta = preguntar_con_reintento(agente, pregunta)
                opciones = [op.strip().lower() for op in esperado.split("/")]
                correcto = any(op in respuesta.lower() for op in opciones)
                if correcto:
                    aciertos += 1
                print(f"  {'OK' if correcto else 'FAIL'} | {pregunta}")
                if not correcto:
                    print(f"       Esperado: {esperado}")
                    print(f"       Obtenido: {respuesta[:120]}")
            except Exception as e:
                errores += 1
                print(f"  ERROR | {pregunta}: {e.__class__.__name__}")

            time.sleep(DELAY)

        total_validas = len(pares) - errores
        precision = (aciertos / total_validas * 100) if total_validas > 0 else 0
        resultados[tipo] = {
            "aciertos": aciertos,
            "total": len(pares),
            "errores": errores,
            "precision": precision,
        }
        print(f"\n  Precision {tipo}: {aciertos}/{total_validas} ({precision:.0f}%) [{errores} errores]")

    return {"modelo": modelo, "resultados": resultados}


PROMPT_CONCISO = """Eres un asistente SQL. Responde preguntas sobre una base de datos SQLite.
1) Lista las tablas, 2) Mira el esquema, 3) Ejecuta la consulta, 4) Responde.
Limita a 5 resultados. No modifiques datos."""


if __name__ == "__main__":
    todos = []

    todos.append(evaluar("gpt-4o-mini"))

    print("\n--- Pausa de 10s entre evaluaciones ---")
    time.sleep(10)

    todos.append(evaluar("gpt-4o-mini", prompt=PROMPT_CONCISO))

    with open("eval_results.json", "w") as f:
        json.dump(todos, f, indent=2, default=str)
    print("\nResultados guardados en eval_results.json")