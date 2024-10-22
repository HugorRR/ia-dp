import streamlit as st
from Assistente_Pessoal import executar

def login_page():
    """
    Função principal para inicializar e executar a aplicação Streamlit.
    """
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if not st.session_state.logged_in:
        st.subheader("Login",divider=True)
        login = st.text_input("Usuario")
        senha = st.text_input("Senha", type="password")
        if st.button("Entrar"):
            if login == "admin" and senha == "admin123":
                st.session_state.logged_in = True
                st.success("Login bem-sucedido!")
                st.rerun()
            else:
                st.error("Login ou senha inválidos")
    else:
        executar()

if __name__ == '__main__':
    login_page()


