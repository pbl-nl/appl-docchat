# logo in user interface
APP_LOGO = "./images/nmdc_logo.png"
# content for application explanation in user interface
APP_INFO="./info/explanation.txt"
# header in user interface
APP_HEADER = "ChatNMDC: chat with your documents"
# folder location of where input documents
DOC_DIR = "./docs"
# folder location of vector database
VECDB_DIR = "./vector_stores"
# folder location of evaluation result
EVAL_DIR = "./evaluate"
# filename of json file with question and answer lists
EVAL_FILE_NAME = "eval.json"
# CHAIN_VERBOSITY must be boolean. True shows standalone question that is conveyed to LLM
CHAIN_VERBOSITY = False

#### the settings below can be used for testing ####
# LLM_TYPE must be one of: "chatopenai", 
LLM_TYPE = "chatopenai"
# LLM_MODEL_TYPE must be one of: "gpt35", "gpt35_16", "gpt4", 
# Context window sizes are currently:
# "gpt35": 4097 tokens which is equivalent to ~3000 words
# "gpt35_16": 16385 tokens
# "gpt4": 8192 tokens
LLM_MODEL_TYPE = "gpt35"
# EMBEDDINGS_PROVIDER must be one of: "openai", 
EMBEDDINGS_PROVIDER = "openai"
# EMBEDDINGS_MODEL must be one of: "text-embedding-ada-002", 
EMBEDDINGS_MODEL = "text-embedding-ada-002"
# CHAIN must be one of: "conversationalretrievalchain", 
CHAIN = "conversationalretrievalchain"
# CHAIN_TYPE must be one of: "stuff", 
CHAIN_TYPE = "stuff"
# SEARCH_TYPE must be one of: "similarity", 
SEARCH_TYPE = "similarity"
# VECDB_TYPE must be one of: "chromadb", 
VECDB_TYPE = "chromadb"
# CHUNK_SIZE must be integer
CHUNK_SIZE = 1000
# CHUNK_OVERLAP must be integer
CHUNK_OVERLAP = 200
# CHUNK_K must be integer (>=1)
CHUNK_K = 4

