# Chat with your docs!
A setup for further exploration of chatting to company documents

## How to use this repo
This repo is created to run in Windows

### Environment setup
1. Open an Anaconda prompt or other command prompt
2. When using conda environments, create a Python environment with conda using commandline command<br>
<code>conda env create -f chatpbl.yml</code><br>
NB: The name of the environment to be created can be changed in the first line of the yml file
3. Activate this environment using commandline command<br>
<code>conda activate chatpbl</code>

### Ingesting documents
The file ingest.py can be used to vectorize all documents in a chosen folder and store the vectors and texts in a vector database for later use.<br>
Execution is done using commandline command:<br>
<code>python ingest.py</code>

### Querying documents
The file query.py can be used to query any folder with documents, provided that the associated vector database exists.<br>
Execution is done using commandline command:<br>
<code>python query.py</code>

### Ingesting and querying documents through a user interface
The functionalities described above can also be used through a User Interface.<br>
The UI can be started by using commandline command:<br>
<code>streamlit run streamlit_app.py</code><br>
When this command is used, a browser session will open automatically

## Suggestions for further development
todo







## References
This repo is mainly inspired by:
- https://blog.langchain.dev/tutorial-chatgpt-over-your-data/
- https://github.com/PatrickKalkman/python-docuvortex/tree/master
- https://docs.streamlit.io/
- https://docs.langchain.com/docs/

### Further useful resources:
todo
