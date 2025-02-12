from textwrap import dedent

OPENAI_RAG_TEMPLATE = dedent("""You are an assistant for question-answering tasks. Use the following pieces of 
    retrieved context to answer the question. If you don't know the answer, just say that you don't know.\n
    Question: {question} \n
    Context: {context} \n
    Helpful answer:""")

OPENAI_RAG_CONCISE_TEMPLATE = dedent("""You are an assistant for question-answering tasks. Use the following pieces of 
    retrieved context to answer the question. If you don't know the answer, just say that you don't know.\n
    Use three sentences maximum and keep the answer concise.\n
    Question: {question} \n
    Context: {context} \n
    Helpful answer:""")

OPENAI_RAG_LANGUAGE_TEMPLATE = dedent("""You are an assistant for question-answering tasks. Use the following pieces of 
    retrieved context to answer the question. If you don't know the answer, just say that you don't know.\n
    Question: {question} \n
    Context: {context} \n
    Answer in the following language: {language}\n
    Helpful answer:""")

SUMMARY_PROMPT_TEMPLATE = dedent(
    """
    "{text}"
    CONCISE SUMMARY:""")

SUMMARY_REFINE_TEMPLATE = dedent(
    """Your job is to produce a final summary.
    We have provided an existing summary up to a certain point: {existing_answer}
    We have the opportunity to refine the existing summary (only if needed) with some more context below.
    ------------
    {text}
    ------------\
    """)

YES_NO_TEMPLATE = dedent("""Use the context below to answer the following question: {question}\n
    Context: {context}\n\n
    Answer the question only with 'yes' or 'no', DO NOT RETURN ANY OTHER TEXT OTHER THAN YES OR NO!\n
    If the context doesn't contain the information to answer the question, the answer will be 'no' """)
