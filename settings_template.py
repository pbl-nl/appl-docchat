"""
This file contains all the settings for the question answering system
"""
# filepath of logo in Streamlit UI
APP_LOGO = "./images/b30.png"
# filepath of text file with content for application explanation in Streamlit UI
APP_INFO = "./info/explanation.txt"
# header in Streamlit UI
APP_HEADER = "CHATPBL"
# filepath of evaluation results folder, e.g. "./evaluation"
EVAL_DIR = "./evaluation"
# header in Streamlit evaluation UI
EVAL_APP_HEADER = "CHATPBL evaluation"
# filepath of text file with content for evaluation explanation in evaluation UI
EVAL_APP_INFO = "./info/evaluation_explanation.txt"
# CHAIN_VERBOSITY must be boolean. When set to True, the standalone question that is conveyed to LLM is shown
CHAIN_VERBOSITY = False
# Location of dotenv file
ENVLOC = "path/to/dotenv/file"


# ######### THE SETTINGS BELOW CAN BE USED FOR TESTING AND CUSTOMIZED TO YOUR PREFERENCE ##########

# TEXT_SPLITTER_METHOD represents the way in which raw text is split into text chunks.
# "RecursiveCharacterTextSplitter" (default): splits text to fixed size chunks
# "NLTKTextSplitter": keeps full sentences even if chunk size is exceeded
TEXT_SPLITTER_METHOD = "NLTKTextSplitter"
# CHUNK_SIZE represents the maximum allowed size of text chunks
CHUNK_SIZE = 1000
# CHUNK_K represents the maximum number of chunks that is conveyed to the LLM context window
# NB: CHUNK_SIZE and CHUNK_K are related, make sure that CHUNK_K * CHUNK_SIZE < LLM context window size
CHUNK_K = 4
# CHUNK_OVERLAP represents the overlap between 2 sequential text chunks, value must be >=0 and < CHUNK_SIZE
CHUNK_OVERLAP = 200

# EMBEDDINGS_PROVIDER must be one of: "openai", "huggingface", "ollama", "azureopenai"
EMBEDDINGS_PROVIDER = "azureopenai"

# - If EMBEDDINGS_PROVIDER is "openai", EMBEDDINGS_MODEL must be one of:
#   "text-embedding-ada-002" (1536 dimensional, max 8191 tokens),
#   "text-embedding-3-small" (1536 dimensional, max 8191 tokens),
#   "text-embedding-3-large" (3072 dimensional, max 8191 tokens)
# - If EMBEDDINGS_PROVIDER is "huggingface", EMBEDDINGS_MODEL must be one of: "all-mpnet-base-v2"
# - If EMBEDDINGS_PROVIDER is "ollama", EMBEDDINGS_MODEL must be one of the locally downloaded models, e.g.
#   "llama3", "nomic-embed-text"
# - EMBEDDINGS_PROVIDER is "azureopenai":

# EMBEDDINGS_MODEL must be one of the embedding models deployed, e.g.
#   "text-embedding-ada-002" (1536 dimensional, max 8191 tokens, cost: $0.10 per 1M tokens),
#   "text-embedding-3-large" (3072 dimensional, max 8191 tokens, cost: $0.13 per 1M tokens)
EMBEDDINGS_MODEL = "text-embedding-ada-002"

# SEARCH_TYPE must be one of: "similarity", "similarity_score_threshold"
# - "similarity": retrieves chunks based on similarity search
# - "similarity_score_threshold": retrieves chunks based on similarity search and a score threshold
# "similarity_score_threshold" is less useful when reranker is used
SEARCH_TYPE = "similarity_score_threshold"

# SCORE_THRESHOLD represents the similarity value that chunks must exceed to qualify for the context.
# It is useful for preventing irrelevant context. Value must be between 0.0 and 1.0
# This value is only relevant when SEARCH_TYPE has been set to "similarity_score_threshold"
# For embedding model text-embedding-ada-002, a value of 0.75 is reasonable
# For embedding model text-embedding-3-large, a value of 0.5 is reasonable
SCORE_THRESHOLD = 0.75

# AZURE_EMBEDDING_DEPLOYMENT_MAP represents a dictionary of Azure embedding model deployments
# with key the model name and value the deployment name
# Adjust for your own Azure model deployments
AZURE_EMBEDDING_DEPLOYMENT_MAP = {
    "text-embedding-ada-002": "pbl-openai-a-cd-ada",
    "text-embedding-3-large": "pbl-openai-a-cd-3large"
}

# RETRIEVER_TYPE represents the type of retrieval that is used to extract the most relevant chunks from the vectorstore
# - "vectorstore": a purely semantic search is done in the vectorstore
# - "hybrid": result will be a combination of vectorstore semantic search and keyword search
# - "parent": semantic search on small chunks but retrieval of larger "parent" chunks
RETRIEVER_TYPE = "vectorstore"

# TEXT_SPLITTER_METHOD_CHILD represents the way in which raw text chunks for the "child" chunks are created.
# Must be one of:
# "RecursiveCharacterTextSplitter": split text to fixed size chunks
# "NLTKTextSplitter": keep full sentences even if chunk size is exceeded
TEXT_SPLITTER_METHOD_CHILD = "NLTKTextSplitter"
# CHUNK_SIZE_CHILD represents the maximum allowed size of "child" chunks, value must be integer
CHUNK_SIZE_CHILD = 200
# CHUNK_K_CHILD represents the number of child chunks that is returned from the vector database.
# Their corresponding parent chunks (number will be <= CHUNK_K_CHILD) are then used as input for the LLM
CHUNK_K_CHILD = 8
# CHUNK_OVERLAP_CHILD represents the overlap between 2 sequential child chunks,
# value must be integer (>=0 and < CHUNK_SIZE_CHILD)
CHUNK_OVERLAP_CHILD = 0

# Only when RETRIEVER_TYPE is se to "hybrid"
# First element represents the weight for BM25 retriever, second element the weight for vectorstore retriever
RETRIEVER_WEIGHTS = [0.7, 0.3]

# MULTIQUERY indicator for whether or not defining multiple queries from users' query
# Value must be False or True
MULTIQUERY = False

# RERANK indicator for whether or not to apply reranking after retrieval
RERANK = True
# RERANK_PROVIDER represents the provider of the reranker model. Must be one of "flashrank_rerank",
RERANK_PROVIDER = "flashrank_rerank"
# RERANK_MODEL represents the reranking model, must be one of the models that are available for download
# from https://huggingface.co/prithivida/flashrank/tree/main
# For more info, see also https://github.com/PrithivirajDamodaran/FlashRank
RERANK_MODEL = "ms-marco-MultiBERT-L-12"
# Number of chunks to retrieve as input for reranker
CHUNK_K_FOR_RERANK = 10

# LLM_PROVIDER must be "openai" in case of using the OpenAI API
# LLM_PROVIDER must be "huggingface" in case of using the Huggingface API
# LLM_PROVIDER must be "ollama" in case of using a downloaded Ollama LLM
# LLM_PROVIDER must be "azureopenai" in case of using the Azure OpenAI Services API
LLM_PROVIDER = "azureopenai"

# - If LLM_PROVIDER is "openai", LLM_MODEL must be one of:
#   "gpt-3.5-turbo", context window size = 4097 tokens (equivalent to ~3000 words)
#   "gpt-3.5-turbo-16k", context window size = 16385 tokens
#   "gpt-4", context window size = 8192 tokens
#   "gpt-4o", context window size = 128000 tokens
# The default llm is "gpt-3.5-turbo"
# - if LLM_PROVIDER is "huggingface", LLM_MODEL should be one of the models as defined on Huggingface,
#   e.g. one of: "meta-llama/Llama-2-7b-chat-hf", "google/flan-t5-base"
#   See Huggingface documentation for context window sizes
#   note: "meta-llama/Llama-2-7b-chat-hf" requires Huggingface Pro Account and access to the llama2 model,
#   see https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
# - If LLM_PROVIDER is "ollama", LLM_MODEL must be one of your Ollama downloaded models, e.g.
#   "llama3"
#   "orca-mini"
#   "zephyr"
#   See also https://ollama.ai/library
# - LLM_PROVIDER is "azureopenai":
#   "gpt-4", context window size = 8192 tokens, cost: $30.00 / $60.00 per 1M tokens

# LLM_MODEL must be one of the LLM models deployed, e.g.:
#   "gpt-35-turbo", context window size = 4097 tokens, cost: $0.50 / $1.50 per 1M tokens
#   "gpt-4o", context window size = 128000 tokens, cost: $2.50 / $10.00 per 1M tokens
LLM_MODEL = "gpt-4o"

# AZURE_LLM_DEPLOYMENT_MAP represents a dictionary of Azure LLM model deployments
# with key the model name and value the deployment name\
# Adjust for your own Azure model deployments
AZURE_LLM_DEPLOYMENT_MAP = {
    "gpt-35-turbo": "pbl-openai-a-cd-openai",
    "gpt-4": "pbl-openai-a-cd-openai4",
    "gpt-4o": "pbl-openai-a-cd-openai4o"
}

# AZURE_OPENAI_ENDPOINT represents the Azure OpenAI endpoint used for connecting to Azure OpenAI API
# This setting is only relevant when EMBEDDINGS_PROVIDER = "azureopenai" or LLM_PROVIDER = "azureopenai"
AZURE_OPENAI_ENDPOINT = "your_azure_openai_endpoint"

# AZURE_OPENAI_API_VERSION represents the Azure OpenAI API version
# This setting is only relevant when EMBEDDINGS_PROVIDER = "azureopenai" or LLM_PROVIDER = "azureopenai"
AZURE_OPENAI_API_VERSION = "your_azure_openai_api_version"

# Similar settings as above, but specifically for evaluation
# EVALUATION_EMBEDDINGS_PROVIDER must be one of "openai", "azureopenai"
EVALUATION_EMBEDDINGS_PROVIDER = "azureopenai"
# - If EVALUATION_EMBEDDINGS_PROVIDER is "openai", EVALUATION_EMBEDDINGS_MODEL must be one of:
#   "text-embedding-ada-002", "text-embedding-3-small" or "text-embedding-3-large"
# - If EVALUATION_EMBEDDINGS_PROVIDER is "azureopenai", EVALUATION_EMBEDDINGS_MODEL must be one of the embedding
#   models deployed, e.g. "text-embedding-ada-002" or "text-embedding-3-large"
EVALUATION_EMBEDDINGS_MODEL = "text-embedding-ada-002"
# EVALUATION_LLM_PROVIDER must be one of "openai", "azureopenai"
EVALUATION_LLM_PROVIDER = "azureopenai"
# - If EVALUATION_LLM_PROVIDER is "openai", EVALUATION_LLM_MODEL must be one of:
#   "gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-4" or "gpt-4o"
# - If EVALUATION_LLM_PROVIDER is "azureopenai", EVALUATION_LLM_MODEL must be one of the LLM models deployed, e.g.:
#   "gpt-35-turbo", "gpt-4" or "gpt-4o"
EVALUATION_LLM_MODEL = "gpt-4o"

# Similar settings as above, but specifically for creation of document summaries
SUMMARY_TEXT_SPLITTER_METHOD = "RecursiveCharacterTextSplitter"
SUMMARY_CHUNK_SIZE = 8000
SUMMARY_CHUNK_OVERLAP = 0
SUMMARY_LLM_PROVIDER = "azureopenai"
SUMMARY_EMBEDDINGS_PROVIDER = "azureopenai"
SUMMARY_EMBEDDINGS_MODEL = "text-embedding-ada-002"
# !! With the langchain version in appl-docchat.yaml it is necessary to choose gpt-35-turbo
# if SUMMARY_LLM_PROVIDER = "azureopenai"
SUMMARY_LLM_MODEL = "gpt-35-turbo"

# settings for confidential documents, using Ollama LLM and embedding model
PRIVATE_LLM_PROVIDER = "ollama"
PRIVATE_LLM_MODEL = "zephyr"
PRIVATE_EMBEDDINGS_PROVIDER = "ollama"
PRIVATE_EMBEDDINGS_MODEL = "nomic-embed-text"
PRIVATE_SUMMARY_LLM_MODEL = "zephyr"

# CHAIN_NAME must be one of: "conversationalretrievalchain",
CHAIN_NAME = "conversationalretrievalchain"

# CHAIN_TYPE must be one of: "stuff",
CHAIN_TYPE = "stuff"

# RETRIEVER_PROMPT represents the type of retriever that is used to extract chunks from the vectorstore
# value must be one of "openai_rag", "openai_rag_concise", "openai_rag_language", "yesno"
# see file prompt_templates.py for explanation
RETRIEVER_PROMPT_TEMPLATE = "openai_rag"

# MAX_INGESTION_SIZE represents the maximum size of the folder in MB
MAX_INGESTION_SIZE = 50
