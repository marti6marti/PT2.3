"""
CLI simple para interactuar con el agente SQL.
Paso 1: python ui.py
Paso 2: python ui.py --v2
"""

import sys
from agents import crear_agente, crear_agente_v2, preguntar
from langgraph.types import Command


def cli_paso1():
    """CLI sin memoria ni human-in-the-loop."""
    print("=== Agente Text2SQL - Paso 1 ===")
    print("Escribe 'salir' para terminar\n")

    agente = crear_agente()

    while True:
        pregunta = input("Pregunta> ").strip()
        if not pregunta:
            continue
        if pregunta.lower() in ("salir", "exit"):
            break

        try:
            respuesta = preguntar(agente, pregunta)
            print(f"\n{respuesta}\n")
        except Exception as e:
            print(f"\nError: {e}\n")


def cli_paso2():
    """CLI con memoria y human-in-the-loop."""
    print("=== Agente Text2SQL - Paso 2 ===")
    print("(Con memoria y human-in-the-loop)")
    print("Escribe 'salir' para terminar\n")

    agente = crear_agente_v2()
    config = {"configurable": {"thread_id": "1"}}

    while True:
        pregunta = input("Pregunta> ").strip()
        if not pregunta:
            continue
        if pregunta.lower() in ("salir", "exit"):
            break

        try:
            # Enviar pregunta al agente
            resultado = agente.invoke(
                {"messages": [{"role": "user", "content": pregunta}]},
                config=config,
            )

            # Bucle de human-in-the-loop
            while True:
                estado = agente.get_state(config)

                # Si no hay interrupciones pendientes, mostrar respuesta
                if not estado.next:
                    ultimo_msg = estado.values["messages"][-1]
                    print(f"\n{ultimo_msg.content}\n")
                    break

                # Hay una herramienta pendiente: mostrar y pedir confirmacion
                ultimo_msg = estado.values["messages"][-1]
                for tool_call in ultimo_msg.tool_calls:
                    print(f"\n  Herramienta: {tool_call['name']}")
                    print(f"  Argumentos: {tool_call['args']}")

                decision = input("\n  Aprobar? (s/n)> ").strip().lower()

                if decision in ("s", "si", "y", "yes"):
                    # Continuar ejecucion
                    resultado = agente.invoke(Command(resume="approve"), config=config)
                else:
                    # Cancelar la herramienta
                    resultado = agente.invoke(Command(resume="reject"), config=config)

        except Exception as e:
            print(f"\nError: {e}\n")


if __name__ == "__main__":
    if "--v2" in sys.argv:
        cli_paso2()
    else:
        cli_paso1()
