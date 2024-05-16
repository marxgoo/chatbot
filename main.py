from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
from pinecone import Pinecone ,ServerlessSpec
from langchain.prompts.prompt import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv

class chatbot():
    load_dotenv()
    #preparing documents
    file_path="Banque_FR.pdf"
    loader= PyPDFLoader(file_path)
    documents = loader.load()
    #Splitting document into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter ( chunk_size=1000, chunk_overlap =150)
    docs = text_splitter.split_documents(documents)
    #creating_text embeddings
    openai_api_key = os.getenv('OPENAI_API_KEY')
    model_name ='text-embedding-ada-002'
    embeddings = OpenAIEmbeddings(
      model=model_name,
      openai_api_key=openai_api_key
    )

    #storing embedding in pinecone
    pinecone_api_key = os.getenv('PINECONE_API_KEY')
    pc = Pinecone (api_key = pinecone_api_key)
    index_name = "langchainchatbot"
    if index_name not in pc.list_indexes().names():
       pc.create_index(
         name="langchainchatbot",
         dimension=1536, 
         metric="cosine", 
         spec=ServerlessSpec(
           cloud="aws",
           region="us-east-1"
         ) 
       )
    #store the embedding
    index = pc.Index(docs, embeddings, index_name)
    #defining the retriever
    retriever = index.as_retriever (search_type='similarity', search_kwargs={'k': 2})

    #building the prompt
    custom_template=""" Give the following conversation and a follow up question rephrase the follow up question to be a standalone question 
    chat History:
    {chat_history}
    FollowUpInput:{question}
    standalone question: """
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(custom_template)
    #creating the chatbotchain
    llm_name='gpt-3.5-turbo'
    qa= ConversationalRetrievalChain.from_llm ( 
      ChatOpenAI ( temperature = 0, model= llm_name, openai_api_key=openai_api_key),
      retriever ,
      condense_question_prompt= CONDENSE_QUESTION_PROMPT,
      returns_source_documents = True
    )

    prompt = "Which sentences do you have ?"

    chat_history = []
    qa({"question": prompt, "chat_history": chat_history}) 



