import os
from typing import List, Tuple
import streamlit as st
from loguru import logger
import pandas as pd
# local imports
import settings
import utils as ut


def folderlist_creator(include_all: bool = True):
    """
    Creates a list of folder names
    Evaluation folder names are found in evaluation output files in folder /evaluate
    """
    folders = [f[:-8] for f in os.listdir(os.path.join(settings.EVAL_DIR, "results")) if
               (os.path.isfile(os.path.join(settings.EVAL_DIR, "results", f)) and f.endswith("_agg.tsv"))]
    if (include_all) and (len(folders) > 1):
        folders.insert(0, "All")

    logger.info("Executed evaluation folderlist_creator()")

    return folders


def folder_selector(folders: List[str]) -> Tuple[str, str, str]:
    """
    selects a document folder and creates the asscciated document folder path and vector database path

    Parameters
    ----------
    folders : List[str]
        list of available folders

    Returns
    -------
    Tuple[str, str, str]
        tuple of selected folder name, its path and its vector database
    """
    # Select source folder with docs
    my_folder_names_selected = st.sidebar.multiselect(label="***SELECT ANY / ALL FOLDERS***",
                                                      default="All",
                                                      options=folders)
    logger.info(f"folder_name_selected is now {my_folder_names_selected}")
    # get associated source folder path and vectordb path
    # my_folder_path_selected = os.path.join(settings.EVAL_DIR, "results", my_folder_name_selected)
    logger.info("Executed folder_selector()")

    return my_folder_names_selected


def compose_dataframes_from_all_eval_files(eval_folders: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    creates aggregated and detailed dataframe to show in the evaluation UI

    Parameters
    ----------
    eval_folders : List[str]
        list of folders with evaluation results

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        dataframe with aggregated results and dataframe with detailed results
    """
    found_eval_folder = False
    if eval_folders == ["All"]:
        eval_folders = folderlist_creator(include_all=False)
    for eval_folder in eval_folders:
        eval_agg_file_name = os.path.join(settings.EVAL_DIR, "results", eval_folder + "_agg.tsv")
        eval_file_name = os.path.join(settings.EVAL_DIR, "results", eval_folder + ".tsv")
        if not found_eval_folder:
            df_agg = pd.read_csv(eval_agg_file_name, sep="\t")
            df_det = pd.read_csv(eval_file_name, sep="\t")
            found_eval_folder = True
        else:
            df_agg = pd.concat([df_agg, pd.read_csv(eval_agg_file_name, sep="\t")], axis=0)
            df_det = pd.concat([df_det, pd.read_csv(eval_file_name, sep="\t")], axis=0)
    df_agg = df_agg.sort_values(by="timestamp", ascending=False)

    admin_columns_agg = ["folder", "timestamp", "eval_file"]
    result_columns = ["answer_relevancy", "context_precision", "faithfulness", "context_recall"]
    settings_dict = ut.get_settings_as_dictionary("settings.py")
    settings_columns = list(settings_dict.keys())
    # select only relevant settings from all settings to show in dataframe
    relevant_setting_columns = [setting_column for setting_column in settings_columns if setting_column not in
                                ["EMBEDDINGS_PROVIDER", "VECDBTYPE", "RERANK_PROVIDER", "LLM_PROVIDER",
                                 "SUMMARY_TEXT_SPLITTER_METHOD", "SUMMARY_CHUNK_SIZE", "SUMMARY_CHUNK_OVERLAP",
                                 "SUMMARY_EMBEDDINGS_PROVIDER", "SUMMARY_EMBEDDINGS_MODEL",
                                 "SUMMARY_LLM_PROVIDER", "SUMMARY_LLM_MODEL", "PRIVATE_LLM_PROVIDER",
                                 "PRIVATE_LLM_MODEL", "PRIVATE_EMBEDDINGS_PROVIDER", "PRIVATE_SUMMARY_LLM_MODEL",
                                 "CHAIN_NAME", "CHAIN_TYPE", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_VERSION",
                                 "EVALUATION_EMBEDDINGS_PROVIDER", "EVALUATION_EMBEDDINGS_MODEL",
                                 "EVALUATION_LLM_PROVIDER", "EVALUATION_LLM_MODEL"]]

    # force order of columns for aggregated results
    df_agg = df_agg.loc[:, admin_columns_agg + result_columns + relevant_setting_columns]
    df_det = df_det.sort_values(by="timestamp", ascending=False)

    # force order of columns for detailed results
    admin_columns_det = ["folder", "timestamp", "eval_file", "file"]
    df_det = df_det.loc[:, admin_columns_det +
                        ["user_input", "reference", "response", "retrieved_contexts"]
                        + result_columns]

    return df_agg, df_det


@st.cache_data
def initialize_page():
    """
    Initializes the main page with a page header and app info
    """
    st.header(settings.EVAL_APP_HEADER)
    with st.expander(label="Show explanation of evaluation metrics"):
        # read app explanation from file evaluation_explanation.txt
        with open(file=settings.EVAL_APP_INFO, encoding="utf8") as file:
            eval_explanation = file.read()
        _, cent_co, _ = st.columns(3)
        with cent_co:
            st.image("./images/ragas_metrics.png")
        st.markdown(body=eval_explanation, unsafe_allow_html=True)
    st.divider()
    logger.info("Executed initialize_page()")


def set_page_config():
    """
    sets the page configuration
    """
    st.set_page_config(page_title="Evaluation", page_icon=':books:', layout='wide')
    logger.info("\nExecuted set_page_config()")


# ### MAIN PROGRAM ####
# set page configuration, this is the first thing that needs to be done
set_page_config()

# Initialize page, executed only once per session
initialize_page()

# create list of content folders
evaluation_folders_available = folderlist_creator()
# chosen folder and associated vector database
folder_names_selected = folder_selector(evaluation_folders_available)

df_eval_agg, df_eval_det = compose_dataframes_from_all_eval_files(folder_names_selected)

st.subheader("Aggregated results")
# from https://docs.streamlit.io/knowledge-base/using-streamlit/how-to-get-row-selections
df_eval_agg_select = df_eval_agg.copy()
df_eval_agg_select.insert(0, "Select", False)
# Get dataframe row-selections from user with st.data_editor
df_agg_ed = st.data_editor(
    df_eval_agg_select,
    hide_index=True,
    column_config={"Select": st.column_config.CheckboxColumn(required=True)},
    disabled=df_eval_agg.columns,
)
# store selected rows and use these to create list of timestamps
selected_rows = df_agg_ed[df_agg_ed.Select]
selected_timestamps = list(selected_rows["timestamp"])
# use selected timestamps as filter for detailed results
df_eval_det = df_eval_det.loc[df_eval_det["timestamp"].isin(selected_timestamps), :]

st.subheader("Detailed results")
st.dataframe(df_eval_det, use_container_width=True)
