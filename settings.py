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
# header in evaluation user interface
EVAL_APP_HEADER = "ChatNMDC: evaluation"
# content for evaluation explanation in evaluation user interface
EVAL_APP_INFO="./info/evaluation_explanation.txt"
# filename of json file with question and answer lists
EVAL_FILE_NAME = "eval.json"
# CHAIN_VERBOSITY must be boolean. True shows standalone question that is conveyed to LLM
CHAIN_VERBOSITY = False

#### the settings below can be used for testing ####
# LLM_TYPE must be one of: "chatopenai", "hugging_face"
LLM_TYPE = "hugging_face"
# if LLM_TYPE is "chatopenai" then LLM_MODEL_TYPE must be one of: "gpt35", "gpt35_16", "gpt4"
# if LLM_TYPE is "hugging_face" then LLM_MODEL_TYPE must be one of "llama2", "GoogleFlan"
# "llama2" requires Huggingface Pro Account and access to the llama2 model https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
# note: llama2 is not fully tested, the last step was not undertaken, because no HF Pro account was available for the developer
# Context window sizes are currently:
# "gpt35": 4097 tokens which is equivalent to ~3000 words
# "gpt35_16": 16385 tokens
# "gpt4": 8192 tokens
LLM_MODEL_TYPE = "GoogleFlan"
# EMBEDDINGS_PROVIDER must be one of: "openai", 
EMBEDDINGS_PROVIDER = "openai"
# EMBEDDINGS_MODEL must be one of: "text-embedding-ada-002", 
EMBEDDINGS_MODEL = "text-embedding-ada-002"
# CHAIN must be one of: "conversationalretrievalchain", 
CHAIN_NAME = "conversationalretrievalchain"
# CHAIN_TYPE must be one of: "stuff", 
CHAIN_TYPE = "stuff"
# SEARCH_TYPE must be one of: "similarity", 
SEARCH_TYPE = "similarity"
# VECDB_TYPE must be one of: "chromadb", 
VECDB_TYPE = "chromadb"
# CHUNK_SIZE must be integer
CHUNK_SIZE = 2000
# CHUNK_OVERLAP must be integer
CHUNK_OVERLAP = 200
# CHUNK_K must be integer (>=1)
CHUNK_K = 2

