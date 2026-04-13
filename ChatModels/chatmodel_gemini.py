# from langchain_google_genai import ChatGoogleGenerativeAI
# from dotenv import load_dotenv

# load_dotenv()

# model = ChatGoogleGenerativeAI(
#     model="gemini-pro"
# )

# result = model.invoke("What is the capital of Pakistan?")

# print(f"Answer from gemini is: {result.content}")


import os
from langchain_google_genai import ChatGoogleGenerativeAI

# 1. Set up your API Key
# You can also set this as an environment variable: GOOGLE_API_KEY
os.environ["GOOGLE_API_KEY"] = "KEY"

# 2. Initialize the Model
# Models include: "gemini-1.5-flash" (fast) or "gemini-1.5-pro" (complex tasks)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)

# 3. Invoke the model
response = llm.invoke("Explain the concept of 'Schrödinger's cat' in two sentences.")

# 4. Print the result
print(response.content)