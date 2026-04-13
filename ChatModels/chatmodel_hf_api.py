from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain.llms import huggingface_hub
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="conversational"
)

model = ChatHuggingFace(llm=llm)

print("hello")

result = model.invoke("What is the capital of Pakistan?")
print(f"Capital is: {result.content}")


