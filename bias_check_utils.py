# import required modules
import json
import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.document_loaders import UnstructuredURLLoader, SeleniumURLLoader
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)


# define function to create vector db using FAISS and the OpenAI embeddings API
def make_vector_db(url):
    """
    creates a vectorstore object with text extracted from url source.
    """

    # load credentials
    with open("openai_key.txt") as f:
        openai_key = f.read()

    embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
    loader = SeleniumURLLoader(urls=[url])
    # loader = UnstructuredURLLoader(urls=[url])

    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=0)

    # load and chunk the document
    doc = loader.load()
    split_doc = text_splitter.split_documents(doc)

    vectordb = FAISS.from_documents(
        split_doc,
        embeddings)

    return vectordb


# define function to create a Q+A retriever
def make_retriever(vectordb):
    """
    create a retriever chain for question-answering over a vectorstore index
    """

    # load credential
    with open("openai_key.txt") as f:
        openai_key = f.read()

    # create model instance
    llm = ChatOpenAI(
        model_name='gpt-4',
        temperature=0,
        request_timeout=180,
        openai_api_key=openai_key
    )

    # compose a template for a system message and context to be passed to the chat API
    system_template = """
    you are an assistant designed to identify potential sources of bias in texts
    
    ----------------
    {context}"""

    messages = [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template("{question}")
    ]
    prompt = ChatPromptTemplate.from_messages(messages)
    chain_type_kwargs = {"prompt": prompt}

    # create the Q+A chain
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectordb.as_retriever(),
        chain_type_kwargs=chain_type_kwargs,
        return_source_documents=False)

    return qa


def id_bias(qa):
    query = """
        Identify forms of bias present in this document.
    """
    return qa({"query": query})['result']


score_prompt = """        
        on a scale of 1 to 4, score the likelihood that each of the forms
        of bias identified in the bias check will generate false or misleading
        information, with
            1 being "not likely to generate false or misleading information",
            2 being "moderately likely to generate false or misleading information",
            3 being "likely to generate false or misleading information", and
            4 being "highly likely to generate false or misleading information."
        explain the reason for each ranking.

        Output should be STRICT JSON, containing
            a dictionary containing the website url,
            a list of dictionaries containing the types of biases with their explanations and scores, and
            a dictionary containing frequency counts for each category of likelihood.
        formatted like this:

        [
            {"url": string},
            [
                {
                    "bias_type": string,
                    "score": number,
                    "score_definition": string
                    "explanation": string
                },
            ],
            {
                "score_frequencies":
                    {
                        "1_not_likely": number,
                        "2_moderately_likely": number,
                        "3_likely": number,
                        "4_highly_likely": number
                    }
            }
        ]
"""


# define function to use the Q+A retriever to generate responses
def generate_response(qa, id_results, url):
    query = score_prompt + f"""
      here is the bias check: {id_results}
      here is the url: {url}
    """
    try:
        answer = qa({"query": query})['result']
    # TODO: specify exception handlers
    except:
        answer = "There may be a temporary issue with the GPT-4 server; try your query again. If the problem persists, check back later."

    return answer


# combine the identification and scoring steps for faster (but sometimes less accurate) responses
def generate_response_fast(qa, url):
    query = "Identify forms of bias present in this document" + score_prompt + f"""
      here is the url: {url}
    """
    try:
        answer = qa({"query": query})['result']
    # TODO: specify exception handlers
    except:
        answer = "There may be a temporary issue with the GPT-4 server; try your query again. If the problem persists, check back later."

    return answer


# create table for HTML display
def make_table(json_data):
    data = json.loads(json_data)
    df = pd.DataFrame()
    for bias in data[1]:
        if df.empty:
            df = pd.json_normalize(bias)
        else:
            df = pd.concat([df, pd.json_normalize(bias)])
    df.columns = df.columns.str.replace('_', ' ').str.title()
    table = df.to_html(index=False,
                       justify="left")

    return table
