OPENAI_RAG: str = """You are an assistant for question-answering tasks. Use the following pieces of
 retrieved context to answer the question. If you don't know the answer, just say that you don't know.\n
 Question: {question} \n
 Context: {context} \n
 Answer:"""

OPENAI_RAG_CONCISE: str = """"You are an assistant for question-answering tasks. Use the following pieces
 of retrieved context to answer the question. If you don't know the answer, just say that you don't know.
 Use three sentences maximum and keep the answer concise.\n
Question: {question} \n
Context: {context} \n
Answer:"""

OPENAI_RAG_LANGUAGE: str = """You are an assistant for question-answering tasks. Use the following pieces of
 retrieved context to answer the question. If you don't know the answer, just say that you don't know.\n
 Question: {question} \n
 Context: {context} \n
 Answer in the following language: {language}"""
