from openai import OpenAI
import os
from dotenv import load_dotenv, find_dotenv
import streamlit as st
from pathlib import Path
import pickle
import fitz  
import asyncio
import nest_asyncio
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from datetime import datetime
import nltk
from nltk.tokenize import word_tokenize
from io import BytesIO
import openai

nltk.download('punkt')
nltk.download('punkt_tab')

nest_asyncio.apply()
_ = load_dotenv(find_dotenv())
openai_key = st.secrets["openai"]["api_key"] 
openai.api_key = openai_key

PASTA_MENSAGENS = Path(__file__).parent / 'mensagens'
PASTA_MENSAGENS.mkdir(exist_ok=True)
CACHE_DESCONVERTE = {}
PDFS = Path(__file__).parent / 'contextos'

contexto_pre_definido = """
A B/Palma é uma empresa de consultoria especializada em contabilidade, fundada em 2016.
Informações da empresa:
- Endereço: R. Santos Dumont, 323 - Cambuí, Campinas - SP, 13024-020
- Telefone: (19) 3381-2671
- Site: www.bpalma.com.br
- CNPJ: 23.327.282/0001-31

Como assistente de departamento pessoal da B/Palma, estou aqui para ajudar o departamento pessoal com questões relacionadas às leis trabalhistas, benefícios e obrigações empregadoras.

Diretrizes para Respostas:

Seja claro e conciso.
Forneça exemplos práticos quando possível.
Se não souber a resposta, indique que a informação não está disponível e sugira consultar um especialista.
Faça perguntas de acompanhamento se a dúvida do usuário não estiver clara.
Exemplos de Perguntas:

Quais são os direitos e obrigações de um empregador em relação ao contrato de trabalho?
Como calcular o valor do FGTS para um funcionário com salário variável?
Seja proativo e pergunte ao usuário se ele precisa de mais informações ou exemplos sobre um tópico específico.

"""

def chama_api(modelo, mensagens, temperatura, max_tokens):
    """
    Função para chamar a API da OpenAI e retornar a resposta.

    Args:
        modelo (str): O modelo a ser usado.
        mensagens (list): Lista de mensagens.
        temperatura (float): A temperatura da resposta.
        max_tokens (int): O número máximo de tokens de retorno.

    Returns:
        dict: A resposta da API.
    """
    client = OpenAI(api_key=openai_key)
    return client.chat.completions.create(model=modelo, messages=mensagens, temperature=temperatura, max_tokens=max_tokens)

async def retorna_resposta_modelo(mensagens, openai_key, modelo='gpt-4o-mini', temperatura=0, max_tokens=2000):
    """
    Função assíncrona para retornar a resposta do modelo.

    Args:
        mensagens (list): Lista de mensagens.
        openai_key (str): Chave da API da OpenAI.
        modelo (str): O modelo a ser usado.
        temperatura (float): A temperatura da resposta.
        max_tokens (int): O número máximo de tokens de retorno.

    Returns:
        str: A resposta do modelo.
    """
    try:
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(None, chama_api, modelo, mensagens, temperatura, max_tokens)
        return response.choices[0].message.content if hasattr(response, 'choices') and len(response.choices) > 0 else None
    except Exception as e:
        st.error(f"Erro ao chamar a API: {e}")
        return None

@st.cache_data
def extrair_informacoes(assunto):
    """
    Função para extrair informações do contexto pré-definido.

    Args:
        assunto (str): O assunto a ser extraído.

    Returns:
        str: O conteúdo do contexto do assunto.
    """
    if assunto == 'Departamento pessoal':
        from contextos.pessoal import departamento_pessoal_bpalma
        return departamento_pessoal_bpalma()
    else:
        return None

def extrair_assunto(mensagem):
    """
    Função para extrair o assunto de uma mensagem.

    Args:
        mensagem (str): A mensagem a ser analisada.

    Returns:
        str: O assunto extraído da mensagem.
    """
    palavras_chave = ['Departamento pessoal']
    mensagem = mensagem.lower()
    for palavra in palavras_chave:
        if palavra.lower() in mensagem:
            return palavra
    return 'desconhecido'

def converte_nome_mensagem(nome_mensagem):
    """
    Function to convert the message name to a suitable format.

    Args:
        nome_mensagem (str): The message name.

    Returns:
        str: The converted message name.
    """
    tokens = word_tokenize(nome_mensagem)
    nome_arquivo = ''.join([token for token in tokens if token.isalnum()])
    return nome_arquivo.lower()

def desconverte_nome_mensagem(nome_arquivo):
    """
    Function to convert the file name back to the message name.

    Args:
        nome_arquivo (str): The file name.

    Returns:
        str: The converted message name.
    """
    if nome_arquivo not in CACHE_DESCONVERTE:
        nome_mensagem = ler_mensagem_por_nome_arquivo(nome_arquivo, key='nome_mensagem')
        CACHE_DESCONVERTE[nome_arquivo] = nome_mensagem
    return CACHE_DESCONVERTE[nome_arquivo]

def ler_mensagem_por_nome_arquivo(nome_arquivo, key='mensagem'):
    """
    Função para ler uma mensagem por nome do arquivo.

    Args:
        nome_arquivo (str): O nome do arquivo.
        key (str): A chave do dicionário da mensagem.

    Returns:
        dict: A mensagem lida do arquivo.
    """
    with open(PASTA_MENSAGENS / nome_arquivo, 'rb') as f:
        mensagens = pickle.load(f)
    return mensagens[key]

def retorna_nome_da_mensagem(mensagens):
    """
    Função para retornar o nome da mensagem.

    Args:
        mensagens (list): Lista de mensagens.

    Returns:
        str: O nome da mensagem.
    """
    nome_mensagem = ''
    for mensagem in mensagens:
        if mensagem['role'] == 'user':
            nome_mensagem = mensagem['content'][:30]
            break
    return nome_mensagem

def salvar_mensagens(mensagens, nome_mensagem, assunto):
    """
    Função para salvar as mensagens em um arquivo.

    Args:
        mensagens (list): Lista de mensagens.
        nome_mensagem (str): O nome da mensagem.
        assunto (str): O assunto da mensagem.
    """
    if len(mensagens) == 0:
        return False

    nome_arquivo = converte_nome_mensagem(nome_mensagem)
    arquivo_salvar = {'nome_mensagem': nome_mensagem, 'nome_arquivo': nome_arquivo, 'mensagem': mensagens, 'assunto': assunto}

    with open(PASTA_MENSAGENS / nome_arquivo, 'wb') as f:
        pickle.dump(arquivo_salvar, f)

def ler_mensagens(mensagens, key='mensagem'):
    """
    Função para ler as mensagens de um arquivo.

    Args:
        mensagens (list): Lista de mensagens.
        key (str): A chave do dicionário da mensagem.

    Returns:
        dict: A mensagem lida do arquivo.
    """
    if len(mensagens) == 0:
        return []
    nome_mensagem = retorna_nome_da_mensagem(mensagens)
    nome_arquivo = converte_nome_mensagem(nome_mensagem)
    with open(PASTA_MENSAGENS / nome_arquivo, 'rb') as f:
        mensagens = pickle.load(f)
    return mensagens[key]

def inicializacao():
    """
    Função para inicializar o estado da sessão do Streamlit.
    """
    if 'mensagens' not in st.session_state:
        st.session_state.mensagens = []
    if 'conversa_atual' not in st.session_state:
        st.session_state.conversa_atual = ''
    if 'assunto' not in st.session_state:
        st.session_state.assunto = ''
    if 'texto_extraido' not in st.session_state:
        st.session_state.texto_extraido = ''
    if 'max_tokens' not in st.session_state:
        st.session_state.max_tokens = 2000
    if 'modelo' not in st.session_state:
        st.session_state.modelo = 'gpt-4o-mini'

@st.cache_data
def processar_documento(uploaded_file):
    """
    Função para processar um documento e extrair o texto.

    Args:
        uploaded_file (UploadedFile): O arquivo carregado.

    Returns:
        str: O texto extraído do arquivo.
    """
    texto_extraido = ""

    if uploaded_file.type == "application/pdf":
        pdf_document = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        for pagina in pdf_document:
            texto_pagina = pagina.get_text()
            texto_extraido += texto_pagina

    elif uploaded_file.type == "text/plain":
        texto_extraido = uploaded_file.read().decode("utf-8")

    elif uploaded_file.type == "text/csv":
        csv_data = pd.read_csv(uploaded_file)
        texto_extraido = csv_data.to_string(index=False)

    elif uploaded_file.type in ["application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "application/vnd.ms-excel"]:
        excel_data = pd.read_excel(uploaded_file)
        texto_extraido = excel_data.to_string(index=False)

    return texto_extraido

async def pagina_principal_async():
    """
    Função assíncrona para gerar a página principal do assistente.
    """
    
    st.header('Bem vindo ao assistente de Departamento Pessoal!', divider=True)

    mensagens = st.session_state.get('mensagens', [])

    for mensagem in mensagens:
        chat = st.chat_message(mensagem['role'])
        chat.markdown(mensagem['content'])

    prompt = st.chat_input('Digite a sua duvida')
    if prompt:
        nova_mensagem = {'role': 'user', 'content': prompt}
        chat = st.chat_message(nova_mensagem['role'])
        chat.markdown(nova_mensagem['content'])

        mensagens.append(nova_mensagem)

        chat = st.chat_message('assistant')
        placeholder = chat.empty()
        placeholder.markdown("▌")
        resposta_completa = ''

        texto_extraido = st.session_state.get('texto_extraido')
        if texto_extraido:
            resposta = await retorna_resposta_modelo(mensagens + [{'role': 'system', 'content': texto_extraido}], openai_key, modelo=st.session_state.modelo, max_tokens=st.session_state.max_tokens)
        else:
            resposta = await retorna_resposta_modelo(mensagens + [{'role': 'system', 'content': contexto_pre_definido}], openai_key, modelo=st.session_state.modelo, max_tokens=st.session_state.max_tokens)

        if resposta:
            if isinstance(resposta, str):
                resposta_completa += resposta
            else:
                resposta_completa += resposta.content or ''
        else:
            resposta_completa = "Não foi possível gerar uma resposta."

        if resposta_completa:
            placeholder.markdown(resposta_completa + "▌")
            placeholder.markdown(resposta_completa)

            nova_mensagem_assistente = {'role': 'assistant', 'content': resposta_completa}
            mensagens.append(nova_mensagem_assistente)

            st.session_state['mensagens'] = mensagens

        nome_mensagem = retorna_nome_da_mensagem(mensagens)
        salvar_mensagens(mensagens, nome_mensagem, st.session_state.get('assunto', ''))

def pagina_principal():
    """
    Função para executar a página principal do assistente.
    """
    
    asyncio.run(pagina_principal_async())

def tab_conversas(tab):
    """
    Função para gerar a interface de conversas na aba de conversas.

    Args:
        tab (Tab): A aba de conversas.
    """
    tab.button('➕ Nova conversa', on_click=seleciona_conversa, args=('',), use_container_width=True)
    tab.markdown('')
    conversas = listar_conversas()
    for nome_arquivo in conversas:
        nome_mensagem = desconverte_nome_mensagem(nome_arquivo).capitalize()
        if len(nome_mensagem) == 30:
            nome_mensagem += '...'
        tab.button(nome_mensagem, on_click=seleciona_conversa, args=(nome_arquivo,), disabled=nome_arquivo == st.session_state['conversa_atual'], use_container_width=True)

def listar_conversas():
    """
    Função para listar as conversas salvas.

    Returns:
        list: Lista de nomes de arquivos das conversas.
    """
    conversas = list(PASTA_MENSAGENS.glob('*'))
    conversas = sorted(conversas, key=lambda item: item.stat().st_atime_ns, reverse=True)
    return [c.stem for c in conversas]

def seleciona_conversa(nome_arquivo):
    """
    Função para selecionar uma conversa salva.

    Args:
        nome_arquivo (str): O nome do arquivo da conversa.
    """
    if nome_arquivo == '':
        st.session_state.mensagens = []
        st.session_state.conversa_atual = ''
        st.session_state.assunto = ''
        st.session_state.texto_extraido = ''
    else:
        mensagem = ler_mensagem_por_nome_arquivo(nome_arquivo, key='mensagem')
        st.session_state.mensagens = mensagem
        st.session_state.conversa_atual = nome_arquivo

def modal_pessoal():
    """
    Função para gerar a interface do modal pessoal.
    """
    assunto = st.selectbox("Selecione um assunto pré-definido", ['Departamento pessoal'], key='select_')
    if st.button("Carregar contexto", key='button_pessoal'):
        if assunto:
            texto_extraido = extrair_informacoes(assunto)
            st.session_state.assunto = assunto
            st.session_state.texto_extraido = texto_extraido
            st.success(f"Contexto do assunto '{assunto}' carregado com sucesso.")
            st.text_area("Conteúdo do Contexto", texto_extraido, height=200)

def salvar_feedback(feedback_text):
    """
    Função para salvar o feedback do usuário.

    Args:
        feedback_text (str): O texto do feedback.
    """
    feedback_folder = Path("feedback")
    feedback_folder.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"feedback_{timestamp}.txt"

    with open(feedback_folder / filename, "w") as f:
        f.write(feedback_text)

def configuracao_modelo():
    """
    Função para gerar a interface de configuração do modelo.
    """
    st.header("Configuração do Modelo")
    modelo = st.selectbox("Selecione o modelo", ['gpt-4o-mini', 'gpt-3.5-turbo', 'gpt-4'], key='select_modelo')
    st.session_state.modelo = modelo
    max_tokens = st.number_input("Quantidade máxima de tokens de retorno", min_value=1, max_value=4096, value=2000)
    st.session_state.max_tokens = max_tokens

    feedback_text = st.text_area('Insira sua pergunta junto com a resposta', '')
    feedback = st.radio("A resposta da IA foi útil?", ['Sim', 'Não'])
    if st.button('Enviar Feedback'):
        salvar_feedback(feedback_text)
        st.success("Feedback enviado com sucesso!")

def analisar_site():
    url_site = st.text_input("Digite o URL do site: ")
    if st.button("Analisar"):
        if url_site:
            st.session_state.url_site = url_site
            st.success("URL armazenada com sucesso!")
    


def ui_tabs():
    tab1, tab2, tab3 = st.sidebar.tabs(['Conversas', 'Configuração', 'Configuração do Modelo'])
    tab_conversas(tab1)
    with tab2:
        st.header("Configuração")
        uploaded_file = st.file_uploader("📁 Procurar arquivo",type=["pdf", "txt", "csv", "xlsx"],label_visibility="collapsed")
        if uploaded_file:
            texto_extraido = processar_documento(uploaded_file)
            if texto_extraido:
                st.session_state.texto_extraido = texto_extraido
                st.success("Documento carregado com sucesso")
        modal_pessoal()
        analisar_site()
    with tab3:
        configuracao_modelo()

st.set_page_config(page_title="Assistente", page_icon="🤖")

def iniciar_sistema():
    """
    Função principal para inicializar e executar a aplicação Streamlit.
    """
    inicializacao()
    pagina_principal()
    ui_tabs()

def executar():
    if 'logged_in' not in st.session_state or not st.session_state.logged_in:
        return
    iniciar_sistema()

if __name__ == '__main__':
    
    executar()

     