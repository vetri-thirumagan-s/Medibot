# For document loading
from langchain.document_loaders import DirectoryLoader,PyPDFLoader
# For text processing
from langchain.text_splitter import RecursiveCharacterTextSplitter
# For embeddings
from langchain_huggingface import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv
# For vector database (Pinecone)
from pinecone import ServerlessSpec
from pinecone.grpc import PineconeGRPC as Pinecone
from langchain_pinecone import PineconeVectorStore
# For AI model interaction
from langchain.chat_models import init_chat_model
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core .prompts import ChatPromptTemplate
from pprint import pprint

load_dotenv()

def load_pdf(data):
    loader = DirectoryLoader(data,
                                glob = "*.pdf",
                                loader_cls = PyPDFLoader)
    documents = loader.load()
    return documents

def text_split(extracted_data):
    text_splitter =  RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 20)
    text_chunk = text_splitter.split_documents(extracted_data)
    return text_chunk

extracted_data = load_pdf("..\\Doc\\")
text_chunk = text_split(extracted_data)

embeddings = HuggingFaceEmbeddings(model_name= "sentence-transformers/all-mpnet-base-v2")

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

pc = Pinecone(api_key= PINECONE_API_KEY)
index_name = "medibot"
pc.create_index(
    name = index_name,
    dimension = 768,
    metric = "cosine",
    spec = ServerlessSpec(
        cloud = "aws",
        region = "us-east-1"
    )
)

vectorstore_from_docs = PineconeVectorStore.from_documents(
        text_chunk,
        index_name=index_name,
        embedding=embeddings,
    )

docsearch = PineconeVectorStore.from_existing_index(
    index_name = index_name,
    embedding = embeddings,
)

retriever = docsearch.as_retriever(search_type = "similarity", search_kwargs = {"k":3})

GROQ_API_KEY = os.environ.get('GROQ_API_KEY')
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

model = init_chat_model("gemma2-9b-It", model_provider= "groq")

system_prompt = (
    "You are an assistant for question-answering tasks."
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you   "
    "don't know. Use three sentences maximum and keep the "
    "answer concise. Analyze each question and give response to it"
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate([
    ("system",system_prompt),
    ("human","{input}")
])

question_answer_chain = create_stuff_documents_chain(model, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

response = rag_chain.invoke({"input" : " aviod  meicine fors acne?"})
res = response["answer"]
pprint(res)