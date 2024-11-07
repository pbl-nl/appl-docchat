import os
from typing import List, Tuple
import streamlit as st
from loguru import logger
import pandas as pd
# local imports
import settings
import utils as ut


def folderlist_creator():
    """
    Creates a list of folder names
    Evaluation folder names are found in evaluation output files in folder /evaluate
    """
    folders = [f[:-8] for f in os.listdir(os.path.join(settings.EVAL_DIR, "results")) if
               (os.path.isfile(os.path.join(settings.EVAL_DIR, "results", f)) and f.endswith("_agg.tsv"))]
    logger.info("Executed evaluation folderlist_creator()")

    return folders


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
    for eval_folder in eval_folders:
        eval_agg_file_name = os.path.join(settings.EVAL_DIR, "results", eval_folder + "_agg.tsv")
        eval_file_name = os.path.join(settings.EVAL_DIR, "results", eval_folder + ".tsv")
        if not found_eval_folder:
            df_agg = pd.read_csv(eval_agg_file_name, sep="\t")
            df = pd.read_csv(eval_file_name, sep="\t")
            found_eval_folder = True
        else:
            df_agg = pd.concat([df_agg, pd.read_csv(eval_agg_file_name, sep="\t")], axis=0)
            df = pd.concat([df, pd.read_csv(eval_file_name, sep="\t")], axis=0)
    df_agg = df_agg.sort_values(by="timestamp", ascending=False)

    admin_columns = ["folder", "timestamp", "eval_file"]
    result_columns = ["answer_relevancy", "context_precision", "faithfulness", "context_recall"]
    settings_dict = ut.get_settings_as_dictionary("settings.py")
    settings_columns = list(settings_dict.keys())

    # force order of columns for aggregated results
    df_agg = df_agg.loc[:, admin_columns + result_columns + settings_columns]
    df = df.sort_values(by="timestamp", ascending=False)

    # force order of columns for detailed results
    df = df.loc[:, admin_columns + ["question", "ground_truth", "answer", "contexts"] + result_columns]

    # from https://docs.streamlit.io/knowledge-base/using-streamlit/how-to-get-row-selections
    df_agg_selections = df_agg.copy()
    df_agg_selections.insert(0, "Select", False)
    # Get dataframe row-selections from user with st.data_editor
    df_agg_ed = st.data_editor(
        df_agg_selections,
        hide_index=True,
        column_config={"Select": st.column_config.CheckboxColumn(required=True)},
        disabled=df_agg.columns,
    )
    # store selected rows and use these to create list of timestamps
    selected_rows = df_agg_ed[df_agg_ed.Select]
    selected_timestamps = list(selected_rows["timestamp"])
    # use selected timestamps as filter for detailed results
    df = df.loc[df["timestamp"].isin(selected_timestamps), :]

    return df_agg_ed, df


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

st.subheader("Aggregated results")
df_eval_agg, df_eval = compose_dataframes_from_all_eval_files(evaluation_folders_available)
st.subheader("Detailed results")
st.dataframe(df_eval, use_container_width=True)
