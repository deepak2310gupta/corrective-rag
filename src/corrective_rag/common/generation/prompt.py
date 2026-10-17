from langchain_core.prompts import ChatPromptTemplate


def create_answer_prompt():

    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful tutor. Answer ONLY using the provided context.\n"
                "If the context is empty or insufficient, say: 'I don't know.'\n"
                "Rules:\n"
                "1. Do not invent facts.\n"
                "2. Do not use information that is not present in the context.\n"
                "3. If the context does not contain enough information to answer,\n"
                "   explicitly say that the answer could not be found in the\n"
                "   available sources.\n"
                "4. Give a concise and accurate answer.",
            ),
            ("human", "Question: {question}\n\nContext:\n{context}"),
        ]
    )


def create_rewrite_query_prompt():

    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You rewrite user question into web search query.\n"
                "Rules:\n"
                "- Keep query short and specific.\n"
                "- Do NOT answer the questions.\n",
            ),
            ("human", "Question: {question}"),
        ]
    )


def create_doc_eval_prompt():
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a document relevance evaluator for a Retrieval-Augmented Generation system.\n"
                "You will be given ONE retrieved chunk and a question.\n"
                "Your task is to evaluate whether the provided document can help answer the user's question."
                "Return a relevance score in [0.0, 1.0].\n"
                "- 1.0: chunk alone is sufficient to answer fully/mostly\n"
                "- 0.0: chunk is irrelevant\n"
                "Scoring guidance:\n"
                "0.0 - Completely irrelevant\n"
                "0.1 - 0.3 - Mostly irrelevant\n"
                "0.4 - 0.6 - Partially relevant / ambiguous\n"
                "0.7 - 0.8 - Relevant\n"
                "0.9 - 1.0 - Highly relevant\n"
                "Be conservative with high scores.\n"
                "Also return a short reason.\n"
                "Important:\n"
                "- Evaluate relevance to the question, not writing quality.\n"
                "- Do not assume information that is not present in the document.\n"
                "- A document can be relevant even if it does not completely answer the question.\n"
                "- Do not use outside knowledge.\n"
                "Output JSON only.",
            ),
            ("human", "Question: {question}\n\nChunk:\n{documentChunk}"),
        ]
    )
