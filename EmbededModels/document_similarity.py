from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

# embeding = OpenAIEmbeddings(model='text-embedding-3-small')

# documents = [
#     "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership",
#     "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
#     "Sachin Tendulkar, also known as 'Master Blaster', holds many batting records",
#     "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
#     "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
# ]

# query = "tell me about rohit sharma"

# doc_embedding = embeding.embed_documents(documents)
# query_embedding = embeding.embed_query(query)

# scores = cosine_similarity([query_embedding],doc_embedding)
# index, score = sorted(list(enumerate(scores)),key=lambda x:x[1])[-1]

# print(documents[index])
# print('similarity score is:', score)


from langchain_openai import OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(model="text-embedding-3-large")

documents = [
    "Virat Kohli is an Indian cricketer known for aggressive batting and leadership",
    "MS Dhoni is a former Indian captain known for finishing matches calmly",
    "Sachin Tendulkar is the highest run scorer in cricket history",
    "Rohit Sharma is known for elegant batting and double centuries",
    "Jasprit Bumrah is a fast bowler known for yorkers"
]

query = "tell me about sachiinn tendolkar"

doc_embedding = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)

scores = cosine_similarity([query_embedding], doc_embedding)[0]

index = scores.argmax()

print(documents[index])
print("similarity score:", scores[index])