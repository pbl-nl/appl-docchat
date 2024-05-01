OPENAI_RAG_TEMPLATE = """You are an assistant for question-answering tasks. Use the following pieces of 
    retrieved context to answer the question. If you don't know the answer, just say that you don't know.\n
    Question: {question} \n
    Context: {context} \n
    Helpful answer:"""

OPENAI_RAG_CONCISE_TEMPLATE = """"You are an assistant for question-answering tasks. Use the following pieces of 
    retrieved context to answer the question. If you don't know the answer, just say that you don't know.\n
    Use three sentences maximum and keep the answer concise.\n
    Question: {question} \n
    Context: {context} \n
    Helpful answer:"""

OPENAI_RAG_LANGUAGE_TEMPLATE = """You are an assistant for question-answering tasks. Use the following pieces of 
    retrieved context to answer the question. If you don't know the answer, just say that you don't know.\n
    Question: {question} \n
    Context: {context} \n
    Answer in the following language: {language}\n
    Helpful answer:"""

SUMMARY_MAPREDUCE_TEMPLATE = """Join the following pieces of text first and then write a summary.
    Pieces of text: {chunks_joined}\n
    Be elaborate and write a comprehensible summary. Return only the summary, no other text. 
    Helpful answer:"""

SUMMARY_TEMPLATE = """Summarize the following text: {text}.\n
    Only return the summary, no explanation.
    Helpful answer:"""

REFINE_TEMPLATE = """Given the following summary: {summary}\n
    Refine this summary by adding the following information: {text}\n
    Only return the refined summary, no other text.
    Helpful answer:"""

SYNTHESIZE_PROMPT_TEMPLATE = """For the following question: \n {question} \n
    Please synthesize/summarize the text below. Hereby cluster which papers have things in common and in which way 
    the clusters differ. Finally, summarize the clusters. \n\n
    Answer: {answer_string}"""
