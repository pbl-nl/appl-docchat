# Chat with your docs!
A RAG (Retrieval Augemented Generation) setup for further exploration of chatting to company documents<br><br><br>
![image](https://github.com/pbl-nl/appl-docchat/assets/7226328/8172f7af-58ef-415b-b8c5-b340d3e7e516)

## How to use this repo
This repo is tested on the Windows platform

### Preparation
1. clone this repo to a folder of your choice
2. create a subfolder vector_stores in the root folder of the cloned repo
3. Create a file .env and enter your OpenAI API key in the first line of this file :<br>
OPENAI_API_KEY="sk-....."<br>
Save and close the .env file<br>
* In case you don't have an OpenAI API key yet, you can obtain one here: https://platform.openai.com/account/api-keys
* Click on + Create new secret key
* Enter an identifier name (optional) and click on Create secret key

### Virtual environment setup
1. Open an Anaconda prompt or other command prompt
2. When using conda environments, go to the root folder of the project and create a Python environment with conda using commandline command<br>
<code>conda env create -f appl-docchat.yml</code><br>
NB: The name of the environment to be created can be changed in the first line of the yml file
3. Activate this environment using commandline command<br>
<code>conda activate appl-docchat</code>

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

## References
This repo is mainly inspired by:
- https://blog.langchain.dev/tutorial-chatgpt-over-your-data/
- https://github.com/PatrickKalkman/python-docuvortex/tree/master
- https://docs.streamlit.io/
- https://docs.langchain.com/docs/

### Further useful resources:
todo
