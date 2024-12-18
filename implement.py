"""
ollama pull llama3.1

pip install langchain langchain-community langchain-ollama
pip install chromadb pypdf

Add your pdf and discord token
"""

# Create Vector DB
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain.vectorstores import Chroma

MODEL_NAME = "llama3.1"
PDF_PATH = "hugman_2024_ML_book.pdf"

def create_vector_db():
  loader = PyPDFLoader(PDF_PATH)
  data = loader.load()
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
  texts = text_splitter.split_documents(data)
  db = Chroma.from_documents(
      documents=texts,
      embedding=OllamaEmbeddings(model=MODEL_NAME)
  )
  return db

vector_db = create_vector_db()

# Set up Retriever
from langchain_ollama.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever

QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""
    {question}
    """
)

llm = ChatOllama(model=MODEL_NAME)

retriever = MultiQueryRetriever.from_llm(
    retriever=vector_db.as_retriever(),
    llm=llm,
    prompt=QUERY_PROMPT
)

# Create Chain
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

template = """
Answer the question based ONLY on the following context: {context}
Question: {question}
Answer: Let's think step by step.
"""

prompt = ChatPromptTemplate.from_template(template)

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Chat using discord
import discord
from discord.ext import commands

with open("./token", "r") as f:
    TOKEN = f.read().strip()

bot = commands.Bot(command_prefix='!', intents=discord.Intents.all())

@bot.event
async def on_ready():
    print(f'Logged in as {bot.user}')
    
@bot.command()
async def chat(ctx, *, query=None):
    if query == None:
        await ctx.send('I need a query!')
        return
    
    answer = llm.invoke(query)
    chunks = [answer.content[i:i+2000] for i in range(0, len(answer.content), 2000)]
    for chunk in chunks:
        await ctx.send(chunk)
    
@bot.command()
async def qa(ctx, *, question=None):
    if question == None:
        await ctx.send('I need a question!')
        return
    
    answer = chain.invoke(question)
    chunks = [answer[i:i+2000] for i in range(0, len(answer), 2000)]
    for chunk in chunks:
        await ctx.send(chunk)

bot.run(TOKEN)