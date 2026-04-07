"""
Evaluaciones del agente SQL - Paso 1
Dataset de 20 pares (pregunta, respuesta esperada).
Prueba con varios modelos y system prompts.
"""

import json
from agents import crear_agente, preguntar

# Dataset: 10 simples + 10 complejas
DATASET = {
    "simples": [
        ("Cuantos artistas hay?", "275"),
        ("Cuantas pistas (tracks) hay?", "3503"),
        ("Cuantos empleados hay?", "8"),
        ("Cuantos clientes son de Brasil?", "5"),
        ("Cuantos generos hay?", "25"),
        ("Cuantas facturas hay?", "412"),
        ("Cuantos tipos de media hay?", "5"),
        ("Cuantos albums hay?", "347"),
        ("Cuantas playlists hay?", "18"),
        ("Cual es el email de Nancy Edwards?", "nancy@chinookcorp.com"),
    ],
    "complejas": [
        ("Que artista tiene mas albums?", "Iron Maiden"),
        ("Cual es el pais con mas clientes?", "USA"),
        ("Cuantas pistas son del genero Rock?", "1297"),
        ("Cual es la pista mas larga en milisegundos?", "Occupation"),
        ("Cual es el ingreso total (suma de totales de facturas)?", "2328"),
        ("Que empleado tiene mas clientes asignados?", "Jane"),
        ("Que cliente ha gastado mas dinero?", "Fern"),
        ("Que genero tiene mas ventas?", "Rock"),
        ("Cuantas pistas tiene el album 'Let There Be Rock'?", "8"),
        ("Cual es la duracion media de las pistas en segundos?", "393"),
    ],
}


def evaluar(modelo="gpt-4o-mini", prompt=None):
    """Evalua el agente con un modelo dado."""
    # Si se pasa un prompt custom, lo usamos modificando el agente
    from agents import SYSTEM_PROMPT
    if prompt:
        # Creamos agente con prompt personalizado
        from langchain_openai import ChatOpenAI
        from langchain_community.agent_toolkits import SQLDatabaseToolkit
        from langgraph.prebuilt import create_react_agent
        from agents import db
        llm = ChatOpenAI(model=modelo)
        toolkit = SQLDatabaseToolkit(db=db, llm=llm)
        agente = create_react_agent(model=llm, tools=toolkit.get_tools(), prompt=prompt)
    else:
        agente = crear_agente(modelo)

    print(f"\n{'='*50}")
    print(f"Modelo: {modelo}")
    print(f"{'='*50}")

    resultados = {}

    for tipo, pares in DATASET.items():
        aciertos = 0
        print(f"\n--- {tipo.upper()} ---")

        for pregunta, esperado in pares:
            try:
                respuesta = preguntar(agente, pregunta)
                correcto = esperado.lower() in respuesta.lower()
                if correcto:
                    aciertos += 1
                estado = "OK" if correcto else "FAIL"
                print(f"  {estado} | {pregunta}")
                if not correcto:
                    print(f"       Esperado: {esperado}")
                    print(f"       Obtenido: {respuesta[:100]}")
            except Exception as e:
                print(f"  ERROR | {pregunta}: {e}")

        precision = aciertos / len(pares) * 100
        resultados[tipo] = precision
        print(f"  Precision {tipo}: {aciertos}/{len(pares)} ({precision:.0f}%)")

    return {"modelo": modelo, "resultados": resultados}


# Segundo system prompt para comparar
PROMPT_CONCISO = """Eres un asistente SQL. Responde preguntas sobre una base de datos SQLite.
1) Lista las tablas, 2) Mira el esquema, 3) Ejecuta la consulta, 4) Responde.
Limita a 5 resultados. No modifiques datos."""


if __name__ == "__main__":
    todos = []

    # Evaluar con gpt-4o-mini y prompt por defecto
    todos.append(evaluar("gpt-4o-mini"))

    # Evaluar con gpt-4o-mini y prompt conciso
    todos.append(evaluar("gpt-4o-mini", prompt=PROMPT_CONCISO))

    # Evaluar con gpt-4o y prompt por defecto (descomentar si tienes acceso)
    # todos.append(evaluar("gpt-4o"))

    # Guardar resultados
    with open("eval_results.json", "w") as f:
        json.dump(todos, f, indent=2)
    print("\nResultados guardados en eval_results.json")
