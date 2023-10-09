APP_LOGO = "./images/pbl_logo_bg.png"
APP_INFO="./info/explanation.txt"
APP_HEADER = "ChatNMDC: chat with your documents"
DOC_DIR = "./docs"
VECDB_DIR = "./vector_stores"

# settings that can be used for testing
LLM_TYPE = "chatopenai"
LLM_MODEL_TYPE = "gpt35"                        # must be one of: "gpt35", "gpt35_16", "gpt4"
EMBEDDINGS_PROVIDER = "openai"
EMBEDDINGS_MODEL = "text-embedding-ada-002"
CHAIN = "conversationalretrievalchain"
CHAIN_TYPE = "stuff"
SEARCH_TYPE = "similarity"                      # must be one of: "similarity", 
VECDB_TYPE = "chromadb"                         # must be one of: "chromadb"
CHUNK_SIZE = 1000                               # must be integer
CHUNK_OVERLAP = 200                             # must be integer
CHUNK_K = 4                                     # must be integer (>=1)
