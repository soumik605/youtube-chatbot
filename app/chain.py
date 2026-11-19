from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

load_dotenv()  # reads .env from the current dir or a parent dir


video_id = "xAt1xcC6qfM"

try:

  #STEP 1

  fetched = YouTubeTranscriptApi().fetch(video_id, languages=["en"])
  # print("🚀 ~ fetched:", fetched)
  transcript_list = fetched.to_raw_data()  # [{'text', 'start', 'duration'}, ...]
  transcript = " ".join(chunk["text"] for chunk in transcript_list)
  # print(transcript)




  # STEP 2

  splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
  chunks = splitter.create_documents([transcript])
  # print("🚀 ~ chunks:", chunks)
  # print("🚀 ~ len chunks:", len(chunks))




  # STEP 3

  embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview",
    output_dimensionality=768,  # Suggested: 768, 1536, or 3072 (default)
  )  # picks up GOOGLE_API_KEY from the environment automatically
  vector_stores = FAISS.from_documents(chunks, embeddings)
  # print("🚀 ~ vector_stores:", vector_stores.index_to_docstore_id)




  # STEP 4

  retriever = vector_stores.as_retriever(search_type="similarity", search_kwargs={"k": 4})
  # print("🚀 ~ retriever:", retriever)

  retriever.invoke("What are the problems in india currently ?")





  # STEP 5

  llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=0.2)
  # print("🚀 ~ llm:", llm)

  prompt = PromptTemplate(
    template="""
      You are a helpful assistant.
      Answer only from the provided transcript context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}
    """,
    input_variables=["context", "question"]
  )



  # STEP 6

  def format_docs(retrieved_docs):
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
    return context_text


  parallel_chain = RunnableParallel({
    "context": retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough()
  })

  question = "What are the problems in india currently ?"
  parallel_chain.invoke(question)

  parser = StrOutputParser()
  main_chain = parallel_chain | prompt | llm | parser

  answer = main_chain.invoke(question)
  print("🚀 ~ answer:", answer)
  

except TranscriptsDisabled:
  print("No captions available for this video.")
except Exception as e:
  print(f"ERROR: {type(e).__name__}: {e}")