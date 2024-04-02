def get_prompt(prompt_type: str) -> str:
    if prompt_type == "openai_rag":
        prompt = """You are an assistant for question-answering tasks. Use the following pieces of
         retrieved context to answer the question. If you don't know the answer, just say that you don't know.\n
         Question: {question} \n
         Context: {context} \n
         Answer:"""
    elif prompt_type == "openai_rag_concise":
        prompt = """"You are an assistant for question-answering tasks. Use the following pieces
         of retrieved context to answer the question. If you don't know the answer, just say that you don't know.\n
         Use three sentences maximum and keep the answer concise.\n
         Question: {question} \n
         Context: {context} \n
         Answer:"""
    elif prompt_type == "openai_rag_language":
        prompt = """You are an assistant for question-answering tasks. Use the following pieces of
         retrieved context to answer the question. If you don't know the answer, just say that you don't know.\n
         Question: {question} \n
         Context: {context} \n
         Answer in the following language: {language}"""

    return prompt
