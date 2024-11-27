"""
This file contains all the settings for the question answering system
"""
# filepath of logo in Streamlit UI
APP_LOGO = "./images/b30.png"
# filepath of text file with content for application explanation in Streamlit UI
APP_INFO = "./info/explanation.txt"
# header in Streamlit UI
APP_HEADER = "ChatPBL"
# filepath of folder with input documents, e.g. "./docs"
DOC_DIR = "./docs"
# filepath of folder with chunks, e.g. "./chunks"
CHUNK_DIR = "./chunks"
# folder for persistent vector databases, e.g. "vector_stores"
VECDB_DIR = "vector_stores"
# filepath of evaluation results folder, e.g. "./evaluate"
EVAL_DIR = "./evaluate"
# header in Streamlit evaluation UI
EVAL_APP_HEADER = "ChatPBL: evaluation"
# filepath of text file with content for evaluation explanation in evaluation UI
EVAL_APP_INFO = "./info/evaluation_explanation.txt"
# CHAIN_VERBOSITY must be boolean. When set to True, the standalone question that is conveyed to LLM is shown
CHAIN_VERBOSITY = False


# ######### THE SETTINGS BELOW CAN BE USED FOR TESTING AND CUSTOMIZED TO YOUR PREFERENCE ##########

# TEXT_SPLITTER_METHOD represents the way in which raw text chunks are created, must be one of:
# "RecursiveCharacterTextSplitter" (split text to fixed size chunks) or
# "NLTKTextSplitter" (keep full sentences even if chunk size is exceeded)
TEXT_SPLITTER_METHOD = "NLTKTextSplitter"
# Only when RETRIEVER_TYPE is set to "parent":
# TEXT_SPLITTER_METHOD_CHILD represents the way in which raw text chunks for the "child" chunks are created.
# Must be one of:
# "RecursiveCharacterTextSplitter" (split text to fixed size chunks) or
# "NLTKTextSplitter" (keep full sentences even if chunk size is exceeded)
TEXT_SPLITTER_METHOD_CHILD = "NLTKTextSplitter"

# CHUNK_SIZE represents the maximum allowed size of text chunks, value must be integer
CHUNK_SIZE = 1000
# Only when RETRIEVER_TYPE is set to "parent":
# CHUNK_SIZE_CHILD represents the maximum allowed size of "child" chunks, value must be integer
CHUNK_SIZE_CHILD = 200

# CHUNK_K represents the number of chunks that is returned from the vector database as input for the LLM
# Value must be integer (>=1)
# NB: CHUNK_SIZE and CHUNK_K are related, make sure that CHUNK_K * CHUNK_SIZE < LLM window size
CHUNK_K = 4
# Only when RETRIEVER_TYPE is set to "parent":
# CHUNK_K_CHILD represents the number of child chunks that is returned from the vector database.
# Their corresponding parent chunks (number will be <= CHUNK_K_CHILD) are then used as input for the LLM
CHUNK_K_CHILD = 4

# CHUNK_OVERLAP represents the overlap between 2 sequential text chunks, value must be integer (>=0 and < CHUNK_SIZE)
CHUNK_OVERLAP = 200
# Only when RETRIEVER_TYPE is set to "parent":
# CHUNK_OVERLAP_CHILD represents the overlap between 2 sequential child chunks,
# value must be integer (>=0 and < CHUNK_SIZE_CHILD)
CHUNK_OVERLAP_CHILD = 0

# EMBEDDINGS_PROVIDER must be one of: "openai", "huggingface", "ollama", "azureopenai"
EMBEDDINGS_PROVIDER = "azureopenai"

# - If EMBEDDINGS_PROVIDER is "openai", EMBEDDINGS_MODEL must be one of:
#   "text-embedding-ada-002" (1536 dimensional, max 8191 tokens),
#   "text-embedding-3-small" (1536 dimensional, max 8191 tokens),
#   "text-embedding-3-large" (3072 dimensional, max 8191 tokens)
# - If EMBEDDINGS_PROVIDER is "huggingface", EMBEDDINGS_MODEL must be one of: "all-mpnet-base-v2"
# - If EMBEDDINGS_PROVIDER is "ollama", EMBEDDINGS_MODEL must be one of the locally downloaded models, e.g.
#   "llama3", "nomic-embed-text"
# - If EMBEDDINGS_PROVIDER is "azureopenai", EMBEDDINGS_MODEL must be the embedding models deployed, e.g.
#   "text-embedding-ada-002" (1536 dimensional, max 8191 tokens),
#   "text-embedding-3-large" (3072 dimensional, max 8191 tokens)
EMBEDDINGS_MODEL = "text-embedding-ada-002"

# SEARCH_TYPE must be one of: "similarity", "similarity_score_threshold"
SEARCH_TYPE = "similarity_score_threshold"

# SCORE_THRESHOLD represents the similarity value that chunks must exceed to qualify for the context.
# Value must be between 0.0 and 1.0
# This value is only relevant when SEARCH_TYPE has been set to "similarity_score_threshold"
# When embedding model text-embedding-ada-002 is used, a value of 0.8 is reasonable
# When embedding model text-embedding-3-large is used, a value of 0.5 is reasonable
SCORE_THRESHOLD = 0.8

# AZURE_EMBEDDING_DEPLOYMENT_MAP represents a dictionary of Azure embedding model deployments
# with key the model name and value the deployment name
# Adjust for your own Azure model deployments
AZURE_EMBEDDING_DEPLOYMENT_MAP = {
    "text-embedding-ada-002": "pbl-openai-a-cd-ada",
    "text-embedding-3-large": "pbl-openai-a-cd-3large"
}

# VECDB_TYPE must be one of: "chromadb",
VECDB_TYPE = "chromadb"

# RETRIEVER_TYPE represents the type of retriever that is used to extract chunks from the vectorstore
# Value must be one of:
# - "vectorstore": in case a purely semantic search is done in the vectorstore (dense vectors)
# - "hybrid": in case a hybrid search is done, the result will be a combination of vectorstore semantic search (dense
# vectors) and BM25 keyword search (sparse vectors)
# - "parent": this uses a ParentDocument retriever meaning that small documents are stored in the vector database
# and used for similarity search while the larger "parent" chunks are returned by the retriever
# NB: the creation of the small documents is steered by the parameters TEXT_SPLITTE_METHOD_CHILD, CHUNK_SIZE_CHILD,
# CHUNK_K_CHILD and CHUNK_OVERLAP_CHILD
RETRIEVER_TYPE = "vectorstore"

# Only applicable if RETRIEVER_TYPE has been set to "hybrid"
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
CHUNKS_K_FOR_RERANK = 20

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
# - If LLM_PROVIDER is "azureopenai", LLM_MODEL must be one of the LLM models deployed, e.g.:
#   "gpt-35-turbo", context window size = 4097 tokens
#   "gpt-4", context window size = 8192 tokens
#   "gpt-4o", context window size = 128000 tokens
LLM_MODEL = "gpt-35-turbo"

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
AZURE_OPENAI_ENDPOINT = "https://pbl-openai-a-ca.openai.azure.com/"

# AZURE_OPENAI_API_VERSION represents the Azure OpenAI API version
# This setting is only relevant when EMBEDDINGS_PROVIDER = "azureopenai" or LLM_PROVIDER = "azureopenai"
AZURE_OPENAI_API_VERSION = "2024-05-01-preview"

# Similar settings as above, but specifically for creation of document summaries
SUMMARY_TEXT_SPLITTER_METHOD = "RecursiveCharacterTextSplitter"
SUMMARY_CHUNK_SIZE = 6000
SUMMARY_CHUNK_OVERLAP = 0
SUMMARY_LLM_PROVIDER = "azureopenai"
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
