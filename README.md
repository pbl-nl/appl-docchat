# Chat with your docs! 
A RAG (Retrieval Augmented Generation) setup for further exploration of chatting to company documents<br><br><br>
![image](https://github.com/pbl-nl/appl-docchat/assets/7226328/d30cdb13-2276-4510-aae9-c94fcaeb9f66)


## How to use this repo
! This repo is tested on a Windows platform

### Preparation
1. Clone this repo to a folder of your choice
2. Create a subfolder vector_stores in the root folder of the cloned repo
3. In the root folder, create a file named ".env" and enter your OpenAI API key in the first line of this file in case you want to use the OpenAI API:<br>
OPENAI_API_KEY="sk-....."<br>
4. Save and close the .env file<br>
* If you don't have an OpenAI API key yet, you can obtain one here: https://platform.openai.com/account/api-keys
* Click on + Create new secret key
* Enter an identifier name (optional) and click on Create secret key
5. In case you want to use one of the open source models API's that are available on huggingface:<br>
Enter your Hugging Face API key in the ".env" file :<br>
HUGGINGFACEHUB_API_TOKEN="hf_....."<br>
* If you don't have an Hugging Face API key yet, you can register at https://huggingface.co/join
* When registered and logged in, you can get your API key in your Hugging Face profile settings
6. This repository also allows for using one of the [Ollama](https://ollama.com/) open source models on-premise. You can do this by follwing the steps below:
* In Windows go to "Turn Windows features on or off" and check the features "Virtual Machine Platform" and "Windows Subsystem for Linux"
* Download and install the Ubuntu Windows Subsystem for Linux (WSL) by opening a terminal window and type <code>wsl --install</code>
* In WSL, install Ollama with <code>curl -fsSL https://ollama.com/install.sh | sh</code>
  
### Conda virtual environment setup
1. Open an Anaconda prompt or other command prompt
2. Go to the root folder of the project and create a Python environment with conda with <code>conda env create -f appl-docchat.yml</code><br>
NB: The name of the environment is appl-docchat by default. It can be changed to a name of your choice in the first line of the file appl-docchat.yml
3. Activate this environment with <code>conda activate appl-docchat</code>

### Pip virtual environment setup
1. Open an Anaconda prompt or other command prompt
2. Go to the root folder of the project and create a Python environment with pip with <code>python -m venv venv</code><br>
This will create a basic virtual environment folder named venv in the root of your project folder
NB: The chosen name of the environment is here appl-docchat. It can be changed to a name of your choice
3. Activate this environment with <code>venv\Scripts\activate</code>
4. All required packages can now be installed with <code>pip install -r requirements.txt</code>

### Setting parameters
The file settings_template.py contains all parameters that can be used and needs to be copied to settings.py. In settings.py, fill in the parameter values you want to use for your use case. 
Examples and restrictions for parameter values are given in the comment lines

### nltk.tokenize.punkt module
When the NLTKTextSplitter is used for chunking the documents, it is necessary to download the punkt module of NLTK.<br>
This can be done in the activated environment by starting a Python interactive session: type <code>python</code>.<br>
Once in the Python session, type <code>import nltk</code> + Enter<br>
Then <code>nltk.download('punkt')</code> + Enter

### Ingesting documents
The file ingest.py can be used to vectorize all documents in a chosen folder and store the vectors and texts in a vector database for later use.<br>
Execution is done in the activated virtual environment with <code>python ingest.py</code>

### Querying documents
The file query.py can be used to query any folder with documents, provided that the associated vector database exists.<br>
Execution is done in the activated virtual environment with <code>python query.py</code>

### Querying multiple documents with multiple questions in batch
The file review.py uses the standard question-answer technique but allows you to ask multiple questions to each document in a folder. 
All the results are gathered in a .csv file.<br>
Execution is done in the activated virtual environment with <code>python review.py</code>

### Ingesting and querying documents through a Streamlit User Interface
The functionalities described above can also be used through a User Interface.<br>
In the activated virtual environment, the UI can be started with <code>streamlit run streamlit_app.py</code><br>
When this command is used, a browser session will open automatically

### Ingesting and querying documents through a Flask User Interface __Needs maintenance__
The functionalities described above can also be used through a Flask User Interface.<br>
The flask UI can be started in the activated virtual environment with <code>python flask_app.py</code>
The Flask UI is tailored for future use in production and contains more insight into the chunks (used) and also contains user admin functionality among others.<br>
For a more detailed description and installation, see the readme file in the flask_app folder

### Summarizing documents
The file summarize.py can be used to summarize every file individually in a document folder. Two options for summarization are implemented:
* Map Reduce: this will create a summary in a fast way. The time (and quality) to create a summary depends on the number of centroids chosen. This is a parameter in settings.py<br>
* Refine: this will create a more refined summary, but can take a long time to run, especially for larger documents

Execution is done in the activated virtual environment with <code>python summarize.py</code>. The user will be prompted for the summarization method, either "Map_Reduce" or "Refine"

### Evaluation of Question Answer results
The file evaluate.py can be used to evaluate the generated answers for a list of questions, provided that the file eval.json exists, containing 
not only the list of questions but also the related list of desired answers (ground truth).<br>
Evaluation is done at folder level in the activated virtual environment with <code>python evaluate.py</code><br>
It is also possible to run an evaluation over all folders with <code>python evaluate_all.py</code>

### Monitoring the evaluation results through a Streamlit User Interface
All evaluation results can be viewed by using a dedicated User Interface.<br>
In the activated virtual environment, this evaluation UI can be started with <code>streamlit run streamlit_evaluate.py</code><br>
When this command is used, a browser session will open automatically

## References
This repo is mainly inspired by:
- https://docs.streamlit.io/
- https://docs.langchain.com/docs/
- https://blog.langchain.dev/tutorial-chatgpt-over-your-data/
- https://github.com/PatrickKalkman/python-docuvortex/tree/master
- https://blog.langchain.dev/evaluating-rag-pipelines-with-ragas-langsmith/
- https://github.com/explodinggradients/ragas

