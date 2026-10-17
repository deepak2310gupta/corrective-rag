from langchain_groq import ChatGroq

def create_llm():

    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0
    )

    return llm