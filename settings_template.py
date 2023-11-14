# relative filepath of logo in user interface, e.g. "./images/nmdc_logo.png"
APP_LOGO = ""

# relative filepath of text file with content for application explanation in Streamlit UI, e.g. "./info/explanation.txt"
APP_INFO = ""

# header in Streamlit UI, e.g. "ChatNMDC: chat with your documents"
APP_HEADER = ""

# relative filepath of folder with input documents, e.g. "./docs"
DOC_DIR = ""

# relative filepath of persistent vector databases, e.g. "./vector_stores"
VECDB_DIR = ""

# relative filepath of evaluation results folder, e.g. "./evaluate"
EVAL_DIR = ""

# header in Streamlit evaluation UI, e.g. "ChatNMDC: evaluation"
EVAL_APP_HEADER = ""

# content for evaluation explanation in evaluation user interface, e.g. "./info/evaluation_explanation.txt"
EVAL_APP_INFO = ""

# filename of json file with question and answer lists, e.g. "eval.json"
EVAL_FILE_NAME = ""

# CHAIN_VERBOSITY must be boolean. When set to True, the standalone question that is conveyed to LLM is shown
CHAIN_VERBOSITY = 

#### The settings below can be used for testing and customized to your own preferences ####
# LLM_TYPE must be one of: "chatopenai", "huggingface", "local_llm"
LLM_TYPE = ""

# - LLM_MODEL_TYPE must be one of: "gpt35", "gpt35_16", "gpt4" if LLM_TYPE is "chatopenai"
#   Context window sizes are currently: "gpt35": 4097 tokens (equivalent to ~3000 words), "gpt35_16": 16385 tokens, "gpt4": 8192 tokens
# - LLM_MODEL_TYPE must be one of: "llama2", "GoogleFlan" if LLM_TYPE is "huggingface"
#   "llama2" requires Huggingface Pro Account and access to the llama2 model https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
#   note: llama2 is not fully tested, the last step was not undertaken, because no HF Pro account was available for the developer
#   Context window sizes are currently: "GoogleFlan": ? tokens, "llama2": ? tokens
# - LLM_MODEL_TYPE must be one of the Ollama downloaded models, e.g. "llama2" "mini-orca" or "zephyr". See also https://ollama.ai/library
LLM_MODEL_TYPE = ""

# API_URL must be the URL to your (local) API
# If LLM_TYPE is "local_llm" and model is run on your local machine, API_URL should be "localhost:11434" by default
# If run on Azure virtual machine, use "http://127.0.0.1:11434"
API_URL = ""

# EMBEDDINGS_PROVIDER must be one of: "openai", "huggingface", "local_embeddings"
EMBEDDINGS_PROVIDER = ""

# - EMBEDDINGS_MODEL must be one of: "text-embedding-ada-002" if EMBEDDINGS_PROVIDER is "openai"
# - EMBEDDINGS_MODEL must be one of: "all-mpnet-base-v2" if EMBEDDINGS_PROVIDER is "huggingface"
# - EMBEDDINGS_MODEL must be one of the locally downloaded models, e.g. "llama2" if EMBEDDINGS_PROVIDER is "local_embeddings"
EMBEDDINGS_MODEL = ""

# TEXT_SPLITTER_METHOD represents the way in which raw text chunks are created, must be one of: 
# "RecursiveCharacterTextSplitter" (split text to fixed size chunks) or 
# "NLTKTextSplitter" (keep full sentences even if chunk size is exceeded)
TEXT_SPLITTER_METHOD = ""

# CHAIN_NAME must be one of: "conversationalretrievalchain", 
CHAIN_NAME = ""

# CHAIN_TYPE must be one of: "stuff", 
CHAIN_TYPE = ""

# SEARCH_TYPE must be one of: "similarity", 
SEARCH_TYPE = ""

# VECDB_TYPE must be one of: "chromadb", 
VECDB_TYPE = ""

# CHUNK_SIZE represents the maximum allowed size of text chunks, value must be integer
CHUNK_SIZE = 

# CHUNK_K represents the number of chunks that is returned from the vector database as input for the LLM, value must be integer (>=1)
# NB: CHUNK_SIZE and CHUNK_K are related, make sure that CHUNK_K * CHUNK_SIZE < LLM window size
CHUNK_K = 

# CHUNK_OVERLAP represents the overlap between 2 sequential text chunks, value must be integer
CHUNK_OVERLAP = 

