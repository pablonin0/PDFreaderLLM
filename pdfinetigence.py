from PyPDF2 import PdfReader
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.vectorstores import ElasticVectorSearch, Weaviate


llm = Ollama(model ="phi")

reader = PdfReader('fluidosMott6v2.pdf')

print(reader)

# read data from the file and put them into a variable called raw_text
raw_text = ''
for i, page in enumerate(reader.pages):
    text = page.extract_text()
    raw_text += text

raw_text[:100]


# We need to split the text that we read into smaller chunks so that during information retreival we don't hit the token size limits. 

text_splitter = CharacterTextSplitter(        
    separator = "\n",
    chunk_size = 1000,
    chunk_overlap  = 200,
    length_function = len,
)
texts = text_splitter.split_text(raw_text)


embeddings = OllamaEmbeddings()

docsearch = FAISS.from_texts(texts, embeddings)

docsearch


chain = load_qa_chain(llm, chain_type="stuff")

query = "who are the authors of the article?"
docs = docsearch.similarity_search(query)
chain.run(input_documents=docs, question=query)