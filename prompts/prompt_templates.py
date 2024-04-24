OPENAI_RAG_TEMPLATE = """You are an assistant for question-answering tasks. Use the following pieces of 
         retrieved context to answer the question. If you don't know the answer, just say that you don't know.\n
         Question: {question} \n
         Context: {context} \n
         Answer:"""

OPENAI_RAG_CONCISE_TEMPLATE = """"You are an assistant for question-answering tasks. Use the following pieces of 
         retrieved context to answer the question. If you don't know the answer, just say that you don't know.\n
         Use three sentences maximum and keep the answer concise.\n
         Question: {question} \n
         Context: {context} \n
         Answer:"""

OPENAI_RAG_LANGUAGE_TEMPLATE = """You are an assistant for question-answering tasks. Use the following pieces of 
         retrieved context to answer the question. If you don't know the answer, just say that you don't know.\n
         Question: {question} \n
         Context: {context} \n
         Answer in the following language: {language}"""

SYNTHESIZE_PROMPT_TEMPLATE = """For the question: \n {question} \n
    Please synthesize/summarize the text below. Hereby cluster which papers have things in common and in which way 
    the clusters differ. Finally, summarize the clusters. \n\n
    Answer: {answer_string}
    """
