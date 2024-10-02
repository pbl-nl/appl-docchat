import re
import setuptools


def fetch_package_property(name):
    with open('__init__.py') as initdotpy:
        src = initdotpy.read()
        result = re.search(r'{}\s*=\s*[\'"]([^\'"]*)[\'"]'.format(name), src)
        return result.group(1)


def description():
    return 'Package for live interaction with your documents through the usage of Large Language Models'


def readme():
    with open('README.md') as file:
        return file.read()


setuptools.setup(
    name='appl-docchat',
    version=fetch_package_property('__version__'),
    description=description(),
    long_description=readme(),
    author=fetch_package_property('__author__'),
    packages=setuptools.find_packages(),
    install_requires=[
    "aiohttp==3.9.5",
    "aiosignal==1.3.1",
    "altair==5.3.0",
    "annotated-types==0.7.0",
    "anyio==4.4.0",
    "appdirs==1.4.4",
    "asgiref==3.8.1",
    "attrs==23.2.0",
    "backoff==2.2.1",
    "bcrypt==4.1.3",
    "beautifulsoup4==4.12.2",
    "blinker==1.8.2",
    "build==1.2.1",
    "cachetools==5.3.3",
    "certifi==2024.6.2",
    "chardet==5.2.0",
    "charset-normalizer==3.3.2",
    "chroma-hnswlib==0.7.3",
    "chromadb==0.5.0",
    "click==8.1.7",
    "colorama==0.4.6",
    "coloredlogs==15.0.1",
    "dataclasses-json==0.6.7",
    "datasets==2.19.2",
    "deepdiff==7.0.1",
    "defusedxml==0.7.1",
    "Deprecated==1.2.14",
    "dill==0.3.8",
    "distro==1.9.0",
    "dnspython==2.6.1",
    "email_validator==2.1.1",
    "emoji==2.12.1",
    "fastapi==0.111.0",
    "fastapi-cli==0.0.4",
    "filelock==3.14.0",
    "filetype==1.2.0",
    "flatbuffers==24.3.25",
    "frozenlist==1.4.1",
    "fsspec==2024.3.1",
    "gitdb==4.0.11",
    "GitPython==3.1.43",
    "google-auth==2.30.0",
    "googleapis-common-protos==1.63.1",
    "greenlet==3.0.3",
    "grpcio==1.64.1",
    "h11==0.14.0",
    "httpcore==1.0.5",
    "httptools==0.6.1",
    "httpx==0.27.0",
    "huggingface-hub==0.23.3",
    "humanfriendly==10.0",
    "idna==3.7",
    "importlib_metadata==7.1.0",
    "importlib_resources==6.4.0",
    "intel-openmp==2021.4.0",
    "Jinja2==3.1.4",
    "joblib==1.4.2",
    "jsonpatch==1.33",
    "jsonpath-python==1.0.6",
    "jsonpointer==3.0.0",
    "jsonschema==4.22.0",
    "jsonschema-specifications==2023.12.1",
    "kubernetes==30.1.0",
    "langchain==0.2.3",
    "langchain-community==0.2.4",
    "langchain-core==0.2.5",
    "langchain-openai==0.1.8",
    "langchain-text-splitters==0.2.1",
    "langdetect==1.0.9",
    "langsmith==0.1.77",
    "loguru==0.7.2",
    "lxml==5.2.2",
    "markdown-it-py==3.0.0",
    "MarkupSafe==2.1.5",
    "marshmallow==3.21.3",
    "mdurl==0.1.2",
    "mkl==2021.4.0",
    "mmh3==4.1.0",
    "monotonic==1.6",
    "mpmath==1.3.0",
    "multidict==6.0.5",
    "multiprocess==0.70.16",
    "mypy-extensions==1.0.0",
    "nest-asyncio==1.6.0",
    "networkx==3.3",
    "nltk==3.8.1",
    "numpy==1.26.4",
    "oauthlib==3.2.2",
    "onnxruntime==1.18.0",
    "openai==1.33.0",
    "opentelemetry-api==1.25.0",
    "opentelemetry-exporter-otlp-proto-common==1.25.0",
    "opentelemetry-exporter-otlp-proto-grpc==1.25.0",
    "opentelemetry-instrumentation==0.46b0",
    "opentelemetry-instrumentation-asgi==0.46b0",
    "opentelemetry-instrumentation-fastapi==0.46b0",
    "opentelemetry-proto==1.25.0",
    "opentelemetry-sdk==1.25.0",
    "opentelemetry-semantic-conventions==0.46b0",
    "opentelemetry-util-http==0.46b0",
    "ordered-set==4.1.0",
    "orjson==3.10.4",
    "overrides==7.7.0",
    "packaging==23.2",
    "pandas==2.2.2",
    "pandocfilters==1.5.0",
    "pillow==10.3.0",
    "posthog==3.5.0",
    "protobuf==4.25.3",
    "pyarrow==16.1.0",
    "pyarrow-hotfix==0.6",
    "pyasn1==0.6.0",
    "pyasn1_modules==0.4.0",
    "pydantic==2.7.3",
    "pydantic_core==2.18.4",
    "pydeck==0.9.1",
    "Pygments==2.18.0",
    "PyMuPDF==1.24.5",
    "PyMuPDFb==1.24.3",
    "pypdf==4.2.0",
    "PyPika==0.48.9",
    "pyproject_hooks==1.1.0",
    "pyreadline3==3.4.1",
    "pysbd==0.3.4",
    "python-dateutil==2.9.0.post0",
    "python-docx==1.1.2",
    "python-dotenv==1.0.1",
    "python-iso639==2024.4.27",
    "python-magic==0.4.27",
    "python-multipart==0.0.9",
    "pytz==2024.1",
    "PyYAML==6.0.1",
    "ragas==0.1.9",
    "rank-bm25==0.2.2",
    "rapidfuzz==3.9.3",
    "referencing==0.35.1",
    "regex==2024.5.15",
    "requests==2.32.3",
    "requests-oauthlib==2.0.0",
    "requests-toolbelt==1.0.0",
    "rich==13.7.1",
    "rpds-py==0.18.1",
    "rsa==4.9",
    "safetensors==0.4.3",
    "scikit-learn==1.5.0",
    "scipy==1.13.1",
    "sentence-transformers==3.0.1",
    "setuptools==69.5.1",
    "shellingham==1.5.4",
    "six==1.16.0",
    "smmap==5.0.1",
    "sniffio==1.3.1",
    "soupsieve==2.5",
    "SQLAlchemy==2.0.30",
    "starlette==0.37.2",
    "streamlit==1.35.0",
    "sympy==1.12.1",
    "tabulate==0.9.0",
    "tbb==2021.12.0",
    "tenacity==8.3.0",
    "threadpoolctl==3.5.0",
    "tiktoken==0.7.0",
    "tinycss2==1.2.1",
    "tokenizers==0.19.1",
    "toml==0.10.2",
    "toolz==0.12.1",
    "torch==2.3.1",
    "tornado==6.4.1",
    "tqdm==4.66.4",
    "transformers==4.41.2",
    "typer==0.12.3",
    "typing_extensions==4.12.2",
    "typing-inspect==0.9.0",
    "tzdata==2024.1",
    "ujson==5.10.0",
    "unstructured==0.14.5",
    "unstructured-client==0.23.3",
    "urllib3==2.2.1",
    "uvicorn==0.30.1",
    "watchdog==4.0.1",
    "watchfiles==0.22.0",
    "webencodings==0.5.1",
    "websocket-client==1.8.0",
    "websockets==12.0",
    "wheel==0.43.0",
    "win32-setctime==1.1.0",
    "wrapt==1.16.0",
    "xxhash==3.4.1",
    "yarl==1.9.4",
    "zipp==3.19.2",
]
,
    include_package_data=True,)