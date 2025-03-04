"""
Grid search to tune chunking hyperparameters
"""
import itertools
import evaluate

# define grid of hyperparameters
CHUNK_SIZE_SET = [500, 1000]
CHUNK_OVERLAP_SET = [100, 200]
CHUNK_K_SET = [2, 4]
TEXT_SPLITTER_SET = ["NLTKTextSplitter", "RecursiveCharacterTextSplitter"]
TEXT_CHILD_SPLITTER_SET = ["NLTKTextSplitter", "RecursiveCharacterTextSplitter"]
CHUNK_CHILD_SIZE_SET = [100, 200]
CHUNK_CHILD_OVERLAP_SET = [50, 100]
SEARCH_TYPE_SET = ["similarity_score_threshold", "similarity"]
SCORE_THRESHOLD_SET = [0.6, 0.8]
RETRIEVER_SET = ["vectorstore", "hybrid", "parent"]
RERANK = [True, False]
# BELOW SHOULD BE IN DICTIONARY FORMAT IN LATER VERSIONS
EMBEDDINGS_PROVIDER = ["azureopenai"]
EMBEDDINGS_MODEL = ["text-embedding-ada-002", "text-embedding-3-large"]
LLM_PROVIDER = ["azureopenai"]
LLM_MODEL = ["gpt-35-turbo", "gpt-4", "gpt-4o"]


def text_processing():
    result = []
    # loop over hyperparameters using grid search
    for chunk_size in CHUNK_SIZE_SET:
        for chunk_overlap in CHUNK_OVERLAP_SET:
            for splitter in TEXT_SPLITTER_SET:
                result.append([chunk_size, chunk_overlap, splitter])
    return result


def embeddings():
    result = []
    for embeddings_provider in EMBEDDINGS_PROVIDER:
        for embeddings_model in EMBEDDINGS_MODEL:
            result.append([embeddings_provider, embeddings_model])
    return result


def llm():
    result = []
    for llm_provider in LLM_PROVIDER:
        for llm_model in LLM_MODEL:
            result.append([llm_provider, llm_model])
    return result


def parent():
    result = []
    for splitter_child in TEXT_CHILD_SPLITTER_SET:
        for chunk_size_child in CHUNK_CHILD_SIZE_SET:
            for chunk_overlap_child in CHUNK_CHILD_OVERLAP_SET:
                result.append(["parent", splitter_child, chunk_size_child, chunk_overlap_child])
    return result


def retriever():
    retriever_settings = []
    for retriever_type in RETRIEVER_SET:
        if retriever_type == "parent":
            retriever_settings.extend(parent())
        else:
            retriever_settings.append([retriever_type, None, 1, 0])
    return retriever_settings


def search_score():
    search_score = []
    for search_type in SEARCH_TYPE_SET:
        if search_type == "similarity_score_threshold":
            for score_threshold in SCORE_THRESHOLD_SET:
                search_score.append([search_type, score_threshold])
        else:
            search_score.append([search_type, 0])
    return search_score


def rerank_chunk_k():
    rerank_chunk_k_settings = []
    for rerank in RERANK:
        if rerank:
            rerank_chunk_k_settings.append([rerank, 0])
        else:
            for chunk_k in CHUNK_K_SET:
                rerank_chunk_k_settings.append([rerank, chunk_k])
    return rerank_chunk_k_settings


def check(settings):
    # check if chunk size is larger than overlap
    if settings["chunk_size"] <= settings["chunk_overlap"]:
        return False
    # check if chunk size is larger than overlap
    if settings["chunk_size_child"] <= settings["chunk_overlap_child"]:
        return False
    # check if child chunk size is larger than chunk size
    if settings["chunk_size"] <= settings["chunk_size_child"]:
        return False
    return True


def get_all_combinations(lists):
    return list(itertools.product(*lists))


def flatten(setting):
    return [item for _list in setting for item in _list]


def to_dict(names, settings):
    return dict(zip(names, settings))


def format_check_settings(names, settings):
    return [to_dict(names, flatten(setting)) for setting in settings if check(to_dict(names, flatten(setting)))]


names = ["chunk_size", "chunk_overlap", "splitter", "embeddings_provider", "embeddings_model",
         "llm_provider", "llm_model", "retriever", "splitter_child", "chunk_size_child", "chunk_overlap_child",
         "search_type", "score_threshold", "rerank", "chunk_k"]
text_processing_settings = text_processing()
embeddings_settings = embeddings()
llm_settings = llm()
retriever_settings = retriever()
search_score_settings = search_score()
rerank_chunk_k_settings = rerank_chunk_k()
combined_settings = get_all_combinations([text_processing_settings, embeddings_settings, llm_settings,
                                          retriever_settings, search_score_settings, rerank_chunk_k_settings])
combined_settings = format_check_settings(names, combined_settings)
for settings in combined_settings:
    evaluate.main(**settings)
