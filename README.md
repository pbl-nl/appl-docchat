# Chat with your docs! 
A RAG (Retrieval Augmented Generation) setup for further exploration of chatting to company documents<br><br><br>
![image](https://github.com/pbl-nl/appl-docchat/assets/7226328/d30cdb13-2276-4510-aae9-c94fcaeb9f66)


## How to use this repo
! This repo is tested on a Windows platform

### Preparation
1. Clone this repo to a folder of your choice
2. In a folder of your choice, create a file named ".env"
3. When using the OpenAI API, enter your OpenAI API key in the first line of this file:<br>
OPENAI_API_KEY="sk-....."<br>
* If you don't have an OpenAI API key yet, you can obtain one here: https://platform.openai.com/account/api-keys
* Click on + Create new secret key
* Enter an identifier name (optional) and click on Create secret key
4. When using Azure OpenAI Services, enter the variable AZURE_OPENAI_API_KEY="....." in the .env file<br>
The value of this variable can be found in your Azure OpenAI Services subscription
5. In case you want to use one of the open source models API's that are available on Huggingface:<br>
Enter your Huggingface API key in the ".env" file :<br>
HUGGINGFACEHUB_API_TOKEN="hf_....."<br>
* If you don't have an Huggingface API key yet, you can register at https://huggingface.co/join
* When registered and logged in, you can get your API key in your Huggingface profile settings
6. This repository also allows for using one of the [Ollama](https://ollama.com/) open source models on-premise. You can do this by following the steps below:
* In Windows go to "Turn Windows features on or off" and check the features "Virtual Machine Platform" and "Windows Subsystem for Linux"
* Download and install the Ubuntu Windows Subsystem for Linux (WSL) by opening a terminal window and type <code>wsl --install</code>
* Start WSL by typing opening a terminal and typing <code>wsl</code>, and install Ollama with <code>curl -fsSL https://ollama.com/install.sh | sh</code>
* When you decide to use a local LLM and/or embedding model, make sure that the Ollama server is running by:
  * opening a terminal and typing <code>wsl</code>
  * starting the Ollama server with <code>ollama serve</code>. This makes any downloaded models accessible through the Ollama API
  
### Conda virtual environment setup
1. Open an Anaconda prompt or other command prompt
2. Go to the root folder of the project and create a Python environment with conda with <code>conda env create -f appl-docchat.yml</code><br>
NB: The name of the environment is appl-docchat by default. It can be changed to a name of your choice in the first line of the file appl-docchat.yml
3. Activate this environment with <code>conda activate appl-docchat</code>

### Pip virtual environment setup
1. Open an Anaconda prompt or other command prompt
2. Go to the root folder of the project and create a Python environment with pip with <code>python -m venv venv</code><br>
This will create a basic virtual environment folder named venv in the root of your project folder
NB: The chosen name of the environment folder is here venv. It can be changed to a name of your choice
3. Activate this environment with <code>venv\Scripts\activate</code>
4. All required packages can now be installed with <code>pip install -r requirements.txt</code>
5. If you would like to run unit tests, you need to <code>pip install -e appl-docchat</code> as well.

### Setting parameters
The file settings_template.py contains all parameters that can be used and needs to be copied to settings.py. In settings.py, fill in the parameter values you want to use for your use case. 
Examples and restrictions for parameter values are given in the comment lines

### NLTK
When the NLTKTextSplitter is used for chunking the documents, it is necessary to download the punkt and punkt_tab module of NLTK.<br>
This can be done in the activated environment by starting a Python interactive session: type <code>python</code>.<br>
Once in the Python session, type <code>import nltk</code> + Enter<br>
Then <code>nltk.download('punkt')</code> + Enter<br>
And finally, <code>nltk.download('punkt_tab')</code> + Enter

### FlashRank reranker
This repo allows reranking the retrieved documents from the vector store, by using the FlashRank reranker
The very first use will download and unzip the required model as indicated in settings.py from HuggingFace platform
For more information on the Flashrank reranker, see https://github.com/PrithivirajDamodaran/FlashRank

### Ingesting documents
The file ingest.py can be used to vectorize all documents in a chosen folder and store the vectors and texts in a vector database for later use.<br>
Execution is done in the activated virtual environment with <code>python ingest.py</code>

### Querying documents
The file query.py can be used to query any folder with documents, provided that the associated vector database exists.<br>
Execution is done in the activated virtual environment with <code>python query.py</code>

### Summarizing documents
The file summarize.py can be used to summarize every file individually in a document folder. Two options for summarization are implemented:
* Map Reduce: this will create a summary in a fast way<br>
* Refine: this will create a more refined summary, but can take a long time to run, especially for larger documents

Execution is done in the activated virtual environment with <code>python summarize.py</code>. The user will be prompted for the summarization method

### Ingesting and querying documents through a Streamlit User Interface
The functionalities described above can also be used through a GUI.<br>
In the activated virtual environment, the GUI can be started with <code>streamlit run streamlit_app.py</code><br>
When this command is used, a browser session will open automatically

### Querying multiple documents with multiple questions in batch
The file review.py uses the standard question-answer technique but allows you to ask multiple questions to each document in a folder sequentially. 
* Create a subfolder with the name <B>review</B> in a document folder
* Secondly, add a file with the name <B>questions.txt</B> in the review folder with all your questions. The file expects a header with column names <B>Question Type</B> and <B>Question</B> for example. Then add all your question types ('Initial', or 'Follow Up' when the question refers to the previous question) and questions tab-separated in the following lines. You can find an example in the docs/CAP_nis folder.<br>

Execution is done in the activated virtual environment with <code>python review.py</code>
All the results, including the answers and the sources used to create the answers, are stored in a file result.csv which is also stored in the subfolder <B>review</B>

### For developers: Monitoring the results of the chunking process through a Streamlit User Interface
When parsing files, the raw text is chunked. To see and compare the results of different chunking methods, use the chunks analysis GUI.<br>
In the activated virtual environment, the chunks analysis GUI can be started with <code>streamlit run streamlit_chunks.py</code><br>
When this command is used, a browser session will open automatically

### For developers: Evaluation of Question Answer results
The file evaluate_all.py can be used to evaluate the generated answers for a list of questions, according to a list of parameter combinations, provided that an evaluation json file exists, containing 
not only the list of questions but also the related list of desired answers (ground truth).<br>
Evaluation is done at folder level (one or multiple folders) in the activated virtual environment with <code>python evaluate_all.py</code><br>

### For developers: Monitoring the evaluation results through a Streamlit User Interface
All evaluation results can be viewed by using a dedicated evaluation GUI.<br>
In the activated virtual environment, this evaluation GUI can be started with <code>streamlit run streamlit_evaluate.py</code><br>
When this command is used, a browser session will open automatically

## References
This repo is mainly inspired by:
- https://docs.streamlit.io/
- https://docs.langchain.com/docs/
- https://blog.langchain.dev/tutorial-chatgpt-over-your-data/
- https://github.com/PatrickKalkman/python-docuvortex/tree/master
- https://github.com/PrithivirajDamodaran/FlashRank
- https://blog.langchain.dev/evaluating-rag-pipelines-with-ragas-langsmith/
- https://github.com/explodinggradients/ragas

