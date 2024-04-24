def get_prompt_from_settings(prompt_type: str) -> str:
    if prompt_type == "openai_rag":
        template = """You are an assistant for question-answering tasks. Use the following pieces of 
         retrieved context to answer the question. If you don't know the answer, just say that you don't know.\n
         Question: {question} \n
         Context: {context} \n
         Answer:"""
    elif prompt_type == "openai_rag_concise":
        template = """"You are an assistant for question-answering tasks. Use the following pieces of 
         retrieved context to answer the question. If you don't know the answer, just say that you don't know.\n
         Use three sentences maximum and keep the answer concise.\n
         Question: {question} \n
         Context: {context} \n
         Answer:"""
    elif prompt_type == "openai_rag_language":
        template = """You are an assistant for question-answering tasks. Use the following pieces of 
         retrieved context to answer the question. If you don't know the answer, just say that you don't know.\n
         Question: {question} \n
         Context: {context} \n
         Answer in the following language: {language}"""

    return template


# def get_review_synthesis_prompt() -> str:
#     template = """For the question: \n {question} \n
#     Please synthesize/summarize the text below. Hereby cluster which papers have things in common and in which way 
#     the clusters differ. Finally, summarize the clusters. \n\n
#     Answer: {answer_string}
#     """

#     return template

SYNTHESIZE_PROMPT_TEMPLATE = """For the question: \n {question} \n
    Please synthesize/summarize the text below. Hereby cluster which papers have things in common and in which way 
    the clusters differ. Finally, summarize the clusters. \n\n
    Answer: {answer_string}
    """
