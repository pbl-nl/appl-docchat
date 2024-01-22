import os
import streamlit as st
from PIL import Image
from loguru import logger
# local imports
import settings
import pandas as pd


def folderlist_creator():
    """
    Creates a list of folder names
    Evaluation folder names are found in evaluation output files in folder /evaluate
    """
    folders = [f[:-8] for f in os.listdir(settings.EVAL_DIR) if 
               (os.path.isfile(os.path.join(settings.EVAL_DIR, f)) and f.endswith("_agg.tsv"))]
    logger.info("Executed evaluation folderlist_creator()")
    return folders


def compose_dataframes_from_all_eval_files(eval_folders):
    found_eval_folder = False
    for eval_folder in eval_folders:
        eval_agg_file_name = os.path.join(settings.EVAL_DIR, eval_folder + "_agg.tsv")
        eval_file_name = os.path.join(settings.EVAL_DIR, eval_folder + ".tsv")
        if not found_eval_folder:
            df_eval_agg = pd.read_csv(eval_agg_file_name, sep="\t")
            df_eval = pd.read_csv(eval_file_name, sep="\t")
            found_eval_folder = True
        else:
            df_eval_agg = pd.concat([df_eval_agg, pd.read_csv(eval_agg_file_name, sep="\t")], axis=0)
            df_eval = pd.concat([df_eval, pd.read_csv(eval_file_name, sep="\t")], axis=0)
    df_eval_agg = df_eval_agg.sort_values(by="timestamp", ascending=False)
    df_eval = df_eval.sort_values(by="timestamp", ascending=False)
    # from https://docs.streamlit.io/knowledge-base/using-streamlit/how-to-get-row-selections
    df_eval_agg_selections = df_eval_agg.copy()
    df_eval_agg_selections.insert(0, "Select", False)
    # Get dataframe row-selections from user with st.data_editor
    df_eval_agg_ed = st.data_editor(
        df_eval_agg_selections,
        hide_index=True,
        column_config={"Select": st.column_config.CheckboxColumn(required=True)},
        disabled=df_eval_agg.columns,
    )
    # store selected rows and use these to create list of timestamps
    selected_rows = df_eval_agg_ed[df_eval_agg_ed.Select]
    selected_timestamps = list(selected_rows["timestamp"])
    # use selected timestamps as filter for detailed results
    df_eval = df_eval.loc[df_eval["timestamp"].isin(selected_timestamps), :]
    return df_eval_agg_ed, df_eval



@st.cache_data
def initialize_page():
    """
    Initializes the main page with a page header and app info
    """
    imagecol, headercol = st.columns([0.3, 0.7])
    logo_image = Image.open(settings.APP_LOGO)
    with imagecol:
        st.image(logo_image, width=250)
    with headercol:
        st.header(settings.EVAL_APP_HEADER)
    with st.expander("Show explanation of evaluation metrics"):
        # read app explanation from file evaluation_explanation.txt
        with open(file=settings.EVAL_APP_INFO) as file:
            eval_explanation = file.read()
        _, cent_co, _ = st.columns(3)
        with cent_co:
            st.image("./images/ragas_metrics.png")
        st.markdown(body=eval_explanation, unsafe_allow_html=True)
    st.divider()
    logger.info("Executed initialize_page()")


def set_page_config():
    st.set_page_config(page_title="ChatNMDC evaluation", page_icon=':books:', layout='wide')
    logger.info("\nExecuted set_page_config()")




#### MAIN PROGRAM ####
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
