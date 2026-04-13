from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

text = 'Islamabad is the capital of Pakistan'

documents = [
    "Islamabad is the capital of Pakistan",
    "Karachi is the capital of Sindh",
    "Paris is the capital of France",
]

# vector = embedding.embed_query(text)
vector = embedding.embed_documents(documents)

print(str(vector))