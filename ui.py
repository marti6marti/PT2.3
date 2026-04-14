
import sys
from langgraph.types import Command



def cli_paso1():
    from agents import crear_agente, preguntar

    print("=== Agente Text2SQL ===")
    print("Escribe 'salir' para terminar\n")
    agente = crear_agente()

    while True:
        pregunta = input("Pregunta> ").strip()
        if not pregunta:
            continue
        if pregunta.lower() in ("salir", "exit"):
            break
        try:
            print(f"\n{preguntar(agente, pregunta)}\n")
        except Exception as e:
            print(f"\nError: {e}\n")


def _loop_hitl(agente, config, solo_riesgo=False):
    from agents import es_herramienta_segura

    while True:
        estado = agente.get_state(config)

        if not estado.next:
            print(f"\n{estado.values['messages'][-1].content}\n")
            return

        ultimo = estado.values["messages"][-1]

        if solo_riesgo and es_herramienta_segura(ultimo.tool_calls):
            agente.invoke(Command(resume=True), config=config)
            continue

        for tc in ultimo.tool_calls:
            print(f"\n  Herramienta: {tc['name']}")
            print(f"  Argumentos: {tc['args']}")

        etiqueta = "Operacion de riesgo. Aprobar?" if solo_riesgo else "Aprobar?"
        decision = input(f"\n  {etiqueta} (s/n)> ").strip().lower()

        if decision in ("s", "si", "y", "yes"):
            agente.invoke(Command(resume=True), config=config)
        else:
            agente.invoke(
                Command(resume=[{"role": "user", "content": "Operacion rechazada por el usuario."}]),
                config=config,
            )


def cli_paso2():
    from agents import crear_agente_v2

    print("=== Agente Text2SQL ===")
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
            agente.invoke({"messages": [{"role": "user", "content": pregunta}]}, config=config)
            _loop_hitl(agente, config, solo_riesgo=False)
        except Exception as e:
            print(f"\nError: {e}\n")


def cli_paso3():
    from agents import crear_agente_v3

    print("=== Agente Text2SQL ===")
    print("Escribe 'salir' para terminar\n")
    agente = crear_agente_v3()
    config = {"configurable": {"thread_id": "1"}}

    while True:
        pregunta = input("Pregunta> ").strip()
        if not pregunta:
            continue
        if pregunta.lower() in ("salir", "exit"):
            break
        try:
            agente.invoke({"messages": [{"role": "user", "content": pregunta}]}, config=config)
            _loop_hitl(agente, config, solo_riesgo=True)
        except Exception as e:
            print(f"\nError: {e}\n")


def cli_paso4():
    from agents import crear_agente_v4, resumir_mensajes

    print("=== Agente Text2SQL ===")
    print("Escribe 'salir' para terminar\n")
    agente, llm, max_msgs = crear_agente_v4()
    config = {"configurable": {"thread_id": "1"}}

    while True:
        pregunta = input("Pregunta> ").strip()
        if not pregunta:
            continue
        if pregunta.lower() in ("salir", "exit"):
            break
        try:
            estado = agente.get_state(config)
            if estado.values and "messages" in estado.values:
                update = resumir_mensajes(llm, estado.values["messages"], max_msgs)
                if update:
                    agente.update_state(config, {"messages": update})
                    print("  (memoria resumida)")

            agente.invoke({"messages": [{"role": "user", "content": pregunta}]}, config=config)
            _loop_hitl(agente, config, solo_riesgo=True)
        except Exception as e:
            print(f"\nError: {e}\n")




def streamlit_app():
    import streamlit as st
    from agents import crear_agente_v4, es_herramienta_segura, resumir_mensajes

    st.title("Agente Text2SQL")

    if "agente" not in st.session_state:
        agente, llm, max_msgs = crear_agente_v4()
        st.session_state.agente = agente
        st.session_state.llm = llm
        st.session_state.max_msgs = max_msgs
        st.session_state.config = {"configurable": {"thread_id": "1"}}
        st.session_state.historial = []
        st.session_state.pendiente = False

    agente = st.session_state.agente
    config = st.session_state.config


    for msg in st.session_state.historial:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])


    if st.session_state.pendiente:
        estado = agente.get_state(config)
        ultimo = estado.values["messages"][-1]
        st.warning("Operacion de riesgo pendiente:")
        for tc in ultimo.tool_calls:
            st.code(f"{tc['name']}: {tc['args']}")

        col1, col2 = st.columns(2)
        if col1.button("Aprobar"):
            agente.invoke(Command(resume=True), config=config)
            st.session_state.pendiente = False
            _streamlit_procesar(agente, config)
            st.rerun()
        if col2.button("Rechazar"):
            agente.invoke(
                Command(resume=[{"role": "user", "content": "Operacion rechazada."}]),
                config=config,
            )
            st.session_state.pendiente = False
            estado = agente.get_state(config)
            if not estado.next:
                resp = estado.values["messages"][-1].content
                st.session_state.historial.append({"role": "assistant", "content": resp})
            st.rerun()
        return


    pregunta = st.chat_input("Escribe tu pregunta SQL...")
    if pregunta:
        st.session_state.historial.append({"role": "user", "content": pregunta})

        # Resumir si hay muchos mensajes
        estado = agente.get_state(config)
        if estado.values and "messages" in estado.values:
            update = resumir_mensajes(
                st.session_state.llm, estado.values["messages"], st.session_state.max_msgs
            )
            if update:
                agente.update_state(config, {"messages": update})

        with st.spinner("Pensando..."):
            agente.invoke(
                {"messages": [{"role": "user", "content": pregunta}]},
                config=config,
            )
            _streamlit_procesar(agente, config)


        st.rerun()


def _streamlit_procesar(agente, config):
    import streamlit as st
    from agents import es_herramienta_segura

    while True:
        estado = agente.get_state(config)
        if not estado.next:
            resp = estado.values["messages"][-1].content
            st.session_state.historial.append({"role": "assistant", "content": resp})
            return

        ultimo = estado.values["messages"][-1]
        if es_herramienta_segura(ultimo.tool_calls):
            agente.invoke(Command(resume=True), config=config)
        else:
            st.session_state.pendiente = True
            return



_in_streamlit = False
try:
    import streamlit as st
    _in_streamlit = st.runtime.exists()
except (ImportError, AttributeError):
    pass

if _in_streamlit:
    streamlit_app()
elif __name__ == "__main__":
    if "--v2" in sys.argv:
        cli_paso2()
    elif "--v3" in sys.argv:
        cli_paso3()
    elif "--v4" in sys.argv:
        cli_paso4()
    else:
        cli_paso1()