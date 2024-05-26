from langchain_community.llms import Ollama

llm = Ollama(model ="phi")

response = llm.invoke("tell me joke")

print(response)