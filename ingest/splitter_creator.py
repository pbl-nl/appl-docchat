import langchain.text_splitter as splitter
# local imports
import settings
import utils as ut


class SplitterCreator():
    """
    Splitter class to import into other modules
    """
    def __init__(self, text_splitter_method=None, chunk_size=None, chunk_overlap=None) -> None:
        self.text_splitter_method = settings.TEXT_SPLITTER_METHOD \
            if text_splitter_method is None else text_splitter_method
        self.chunk_size = settings.CHUNK_SIZE if chunk_size is None else chunk_size
        self.chunk_overlap = settings.CHUNK_OVERLAP if chunk_overlap is None else chunk_overlap

    def get_splitter(self):
        """
        Get the text splitter object
        """
        if self.text_splitter_method == "NLTKTextSplitter":
            return lambda language = 'english': splitter.NLTKTextSplitter(
                separator="\n\n",
                language=ut.language_map.get(language,'english'),
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
        elif self.text_splitter_method == "RecursiveCharacterTextSplitter":
            return lambda language = "english": splitter.RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
                chunk_overlap=self.chunk_overlap
            )
