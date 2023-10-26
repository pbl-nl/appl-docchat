# Chat with your docs!
A RAG (Retrieval Augmented Generation) setup for further exploration of chatting to company documents<br><br><br>
![image](https://github.com/pbl-nl/appl-docchat/assets/7226328/d30cdb13-2276-4510-aae9-c94fcaeb9f66)


## How to use this repo
! This repo is tested on a Windows platform

### Preparation
1. Clone this repo to a folder of your choice
2. Create a subfolder vector_stores in the root folder of the cloned repo
3. Create a file .env and enter your OpenAI API key in the first line of this file :<br>
OPENAI_API_KEY="sk-....."<br>
Save and close the .env file<br>
* In case you don't have an OpenAI API key yet, you can obtain one here: https://platform.openai.com/account/api-keys
* Click on + Create new secret key
* Enter an identifier name (optional) and click on Create secret key

### Conda virtual environment setup
1. Open an Anaconda prompt or other command prompt
2. Go to the root folder of the project and create a Python environment with conda using commandline command<br>
<code>conda env create -f appl-docchat.yml</code><br>
NB: The name of the environment is appl-docchat by default. It can be changed to a name of your choice in the first line of the yml file
3. Activate this environment using commandline command<br>
<code>conda activate appl-docchat</code>

### Pip virtual environment setup
1. Open an Anaconda prompt or other command prompt
2. Go to the root folder of the project and create a Python environment with pip using commandline command<br>
<code>python -m venv venv</code><br>
This will create a basic virtual environment folder named venv in the root of your project folder
NB: The chosen name of the environment is here appl-docchat. It can be changed to a name of your choice
3. Activate this environment using commandline command<br>
<code>venv\Scripts\activate</code>
4. All required packages can now be installed with command line command<br>
<code>pip install -r requirements.txt</code>

### Ingesting documents
The file ingest.py can be used to vectorize all documents in a chosen folder and store the vectors and texts in a vector database for later use.<br>
Execution is done in the activated virtual environment using commandline command:<br>
<code>python ingest.py</code>

### Querying documents
The file query.py can be used to query any folder with documents, provided that the associated vector database exists.<br>
Execution is done in the activated virtual environment using commandline command:<br>
<code>python query.py</code>

### Ingesting and querying documents through a user interface
The functionalities described above can also be used through a User Interface.<br>
The UI can be started by using commandline command:<br>
<code>streamlit run streamlit_app.py</code><br>
When this command is used, a browser session will open automatically

### Evaluation of Question Answer results
The file evaluate.py can be used to evaluate the generated answers for a list of questions, provided that the file eval.json exists, containing 
not only the list of questions but also the related list of desired answers (ground truth).<br>
Evaluation is done in the activated virtual environment using commandline command:<br>
<code>python evaluate.py</code>

### Monitoring the evaluation results through a user interface
All evaluation results can be viewed by using a dedicated User Interface.<br>
This evaluation UI can be started by using commandline command:<br>
<code>streamlit run streamlit_evaluate.py</code><br>
When this command is used, a browser session will open automatically

## User Stories for improvements
User stories are divided into 2 groups:<br> 
* RESEARCH: RESEARCH user stories are not meant to change any code but require research and prepare for an actual BUILD task.
* BUILD: BUILD user stories change the code. They add functionality to the application or are performance related
<br>Furthermore, every user story below has an indication whether it would extend the functionality of the application (FUNC), or is related to optimize the results (EVAL).
User stories are written from the perspective of either the user of the application, or the developer of the application.
1. Ingestion (1): As a user I want to synchronize the vector database with the document folder I am using. If the document folder has changed (extra file(s) or deleted file(s)), either add extra documents to the vector database or delete documents from the vector database. FUNC, BUILD
2. Ingestion (2): As a user I want to query not only PDF’s, but also other file types with text, like Word documents, plain text files, and html pages. FUNC, BUILD
3. Ingestion (3): As a developer I want to determine the optimal settings for chunking. Current settings are chunksize = 1000 and chunk overlap = 200 (see settings.py). Can we do some tests with evaluation documents and find an optimal chunk size and overlap? EVAL, BUILD
4. Ingestion (3): As a developer I want to use an optimal set of chunks. Can we implement content-aware text chunking, keeping related content together in one chunk (up to a maximum chunk size)? EVAL, BUILD. For inspiration: https://github.com/nlmatics/llmsherpa#layoutpdfreader
5. Ingestion (4) & Retrieval (7): As a developer I want to generate the best answers to the user questions. The application currently uses OpenAI's text-embedding-ada-002 as embedding model. It is not the best one according to huggingface MTEB embedding leaderboard. See https://huggingface.co/spaces/mteb/leaderboard. Implement an alternative embedding model and evaluate any change in performance EVAL, BUILD
6. Retrieval (9): As a user I don’t want the chatbot to hallucinate. Add a lower bound for the similarity score to filter out text chunks. If none of the text chunks reaches the lower bound value, answer “I don’t know” (in the language of the user)? EVAL, BUILD
7. Retrieval (9): As a user I want to know which returned chunks are the preferred ones. Add the similarity score of each chunk to the sources in the response and rank each chunk according to the similarity score FUNC, BUILD
8. Retrieval (11): As a developer I want to evaluate the impact of switching from LLM gpt 3.5 to gpt 4? EVAL, BUILD
9. 

## References
This repo is mainly inspired by:
- https://docs.streamlit.io/
- https://docs.langchain.com/docs/
- https://blog.langchain.dev/tutorial-chatgpt-over-your-data/
- https://github.com/PatrickKalkman/python-docuvortex/tree/master
- https://blog.langchain.dev/evaluating-rag-pipelines-with-ragas-langsmith/
- https://github.com/explodinggradients/ragas

