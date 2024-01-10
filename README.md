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
4. Enter your Hugging Face API key in the second line of this file :<br>
HUGGINGFACEHUB_API_TOKEN="hf_....."<br>
* In case you don't have an Hugging Face API key yet, you can register at https://huggingface.co/join
* When registered and logged in, you can get your API key in your Hugging Face profile settings

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

### Ingesting and querying documents through a Streamlit User Interface
The functionalities described above can also be used through a User Interface.<br>
The UI can be started by using commandline command:<br>
<code>streamlit run streamlit_app.py</code><br>
When this command is used, a browser session will open automatically

### Ingesting and querying documents through a Flask User Interface
The functionalities described above can also be used through a Flask User Interface.<br>
The flask UI can be started in the activated virtual environment using commandline command:<br>
<code>python flask_app.py</code>
The Flask UI is tailored for future use in production and contains more insight into the chunks (used) and also contains user admin functionality among others.<br>
For a more detailed description and installation, see the readme file in the  flask_app folder

### Evaluation of Question Answer results
The file evaluate.py can be used to evaluate the generated answers for a list of questions, provided that the file eval.json exists, containing 
not only the list of questions but also the related list of desired answers (ground truth).<br>
Evaluation is done at folder level in the activated virtual environment using commandline command:<br>
<code>python evaluate.py</code><br>
It is also possible to run an evaluation over all folders with <code>python evaluate_all.py</code>

### Monitoring the evaluation results through a Streamlit User Interface
All evaluation results can be viewed by using a dedicated User Interface.<br>
This evaluation UI can be started by using commandline command:<br>
<code>streamlit run streamlit_evaluate.py</code><br>
When this command is used, a browser session will open automatically

## References
This repo is mainly inspired by:
- https://docs.streamlit.io/
- https://docs.langchain.com/docs/
- https://blog.langchain.dev/tutorial-chatgpt-over-your-data/
- https://github.com/PatrickKalkman/python-docuvortex/tree/master
- https://blog.langchain.dev/evaluating-rag-pipelines-with-ragas-langsmith/
- https://github.com/explodinggradients/ragas

