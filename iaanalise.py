# RAG com Planilhas Excel Para Análise Financeira com IA Generativa

# Importar pacotes utilizados
import os
import gc
import tempfile
import uuid
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from llama_index.core import Settings
from llama_index.core import PromptTemplate
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.readers.docling import DoclingReader
from llama_index.core.node_parser import MarkdownNodeParser
import streamlit as st
import warnings
warnings.filterwarnings('ignore')
from huggingface_hub import login  # Importe a função login

# Configuração da página com título e ícone
st.set_page_config(page_title="RAG Excel Análise", page_icon=":100:")

# Inicialização de variáveis de sessão
if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()
    st.session_state.file_cache = {}

session_id = st.session_state.id

# Cliente inicializado como None
client = None

# Função para carregar o modelo de linguagem LLM
@st.cache_resource
def dsa_carrega_llm():
    st.write("Carregando modelo LLM...")
    # Autentique-se no Hugging Face antes de carregar o modelo
    login(token="YOUR_HUGGINGFACE_TOKEN")  # Substitua pelo seu token
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    st.write("Modelo LLM carregado com sucesso!")
    return tokenizer, model

# Função para resetar o chat (botão limpar)
def dsa_reset_chat():
    st.session_state.messages = []
    st.session_state.context = None
    gc.collect()

# Função para exibir o conteúdo do Excel (opcional)
def dsa_display_excel(file):
    st.markdown("### Excel Preview")
    df = pd.read_excel(file)
    st.dataframe(df)

# Seção lateral para upload de arquivos
with st.sidebar:

    st.header(f"Adicione seu documento!")

    # Carregamento de arquivo pelo usuário
    uploaded_file = st.file_uploader("Selecione seu arquivo Excel:", type=["xlsx", "xls"])

    if uploaded_file:

        try:
            # Criar um diretório temporário para processar o arquivo
            with tempfile.TemporaryDirectory() as temp_dir:
                file_path = os.path.join(temp_dir, uploaded_file.name)

                # Salvar o arquivo carregado
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())

                file_key = f"{session_id}-{uploaded_file.name}"
                st.write("Indexando o documento...")

                # Verificar se o arquivo já foi processado
                if file_key not in st.session_state.get('file_cache', {}):

                    # Validar se o diretório temporário existe
                    if os.path.exists(temp_dir):
                        reader = DoclingReader()
                        loader = SimpleDirectoryReader(input_dir=temp_dir, file_extractor={".xlsx": reader})
                    else:
                        st.error('Não foi possível encontrar o arquivo que você enviou, verifique novamente...')
                        st.stop()

                    # Carregar os dados do arquivo
                    docs = loader.load_data()

                    # Carregar o modelo LLM
                    tokenizer, model = dsa_carrega_llm()

                    # Configurar modelo de embeddings
                    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5", trust_remote_code=True)

                    # Configurar embeddings no ambiente de configurações
                    Settings.embed_model = embed_model

                    # Criar o parser para os nós do documento
                    node_parser = MarkdownNodeParser()

                    # Criar o índice baseado nos documentos carregados
                    index = VectorStoreIndex.from_documents(documents=docs, transformations=[node_parser], show_progress=True)

                    # Criar o motor de consulta para os dados
                    query_engine = index.as_query_engine(streaming=True)

                    # Definir o template de prompt para consultas
                    qa_prompt_tmpl_str = (
                        "Contexto abaixo.\n"
                        "---------------------\n"
                        "{context_str}\n"
                        "---------------------\n"
                        "Você é um analista financeiro experiente. Use as informações acima para responder à pergunta do usuário da forma mais precisa e concisa possível. Se a pergunta não puder ser respondida com as informações fornecidas, responda 'Não sei'.\n"
                        "Pergunta: {query_str}\n"
                        "Resposta: "
                    )

                    # Configurar o template de prompt
                    qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)

                    # Atualizar prompts no motor de consulta
                    query_engine.update_prompts({"response_synthesizer:text_qa_template": qa_prompt_tmpl})

                    # Salvar o motor de consulta na sessão
                    st.session_state.file_cache[file_key] = query_engine
                else:
                    # Recuperar o motor de consulta já processado
                    query_engine = st.session_state.file_cache[file_key]

                # Exibir mensagem de sucesso e mostrar o conteúdo da planilha
                st.success("Sistema Pronto!")
                dsa_display_excel(uploaded_file)
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.stop()

# Divisão da interface em duas colunas
col1, col2 = st.columns([6, 1])

# Cabeçalho principal
with col1:
    st.header(f"RAG com Planilhas Excel Para Análise Financeira com IA Generativa")

# Botão para limpar o chat
with col2:
    st.button("Limpar", on_click=dsa_reset_chat)

# Linha divisória visual
st.markdown("---")

# Inicializar histórico de mensagens, se não existir
if "messages" not in st.session_state:
    dsa_reset_chat()

# Exibir mensagens anteriores no histórico
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Aceitar entrada do usuário
if prompt := st.chat_input("Digite sua pergunta para analisar a planilha."):

    # Adicionar mensagem do usuário ao histórico
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Exibir a mensagem do usuário no chat
    with st.chat_message("user"):
        st.markdown(prompt)

    # Exibir a resposta do assistente
    with st.chat_message("assistant"):

        # Cria o placeholder vazio
        message_placeholder = st.empty()

        # Inicializa a variável para a resposta completa
        full_response = ""

        # Consultar o motor e processar a resposta em partes
        if "file_cache" in st.session_state and st.session_state.file_cache:
            streaming_response = query_engine.query(prompt)

            # Loop pelos chunks
            for chunk in streaming_response.response_gen:
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")

            # Mostrar a resposta final
            message_placeholder.markdown(full_response)

            # Adicionar resposta do assistente ao histórico
            st.session_state.messages.append({"role": "assistant", "content": full_response})
        else:
            st.error("O sistema não está pronto. Por favor, carregue um arquivo primeiro.")
