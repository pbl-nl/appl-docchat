from typing import Dict, List, Tuple
import re
from loguru import logger
from langchain_community.document_loaders import BSHTMLLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
import fitz
from docx2pdf import convert
import os
from fpdf import FPDF
import re
import pdfkit
# local imports
import utils as ut


class FileParser:
    """
    A class with functionality to parse various kinds of files
    """
    def __init__(self) -> None:
        pass

    def parse_file(self, file_path: str):
        if file_path.endswith(".pdf"):
            raw_pages, metadata = self.parse_pymupdf(file_path)
        elif file_path.endswith(".txt") or file_path.endswith(".md"):
            raw_pages, metadata = self.parse_txt(file_path)
        elif file_path.endswith(".html"):
            raw_pages, metadata = self.parse_html(file_path)
        elif file_path.endswith(".docx"):
            raw_pages, metadata = self.parse_word(file_path)

        # return raw text from pages and metadata
        return raw_pages, metadata

    def get_metadata(self, file_path: str, doc_metadata: str):
        """
        Extracts the following metadata from the pdf document:
        title, author(s) and full filename
        For CLO, indicator_url and indicator_closed are added, they will have no effect for other documents
        """
        return {"title": doc_metadata.get('title', '').strip(),
                "author": doc_metadata.get('author', '').strip(),
                # created specifically for CLO
                "indicator_url": doc_metadata.get('keywords', '').strip(),
                "indicator_closed": doc_metadata.get('trapped', '').strip(),
                # add filename to metadata
                "filename": file_path.split('\\')[-1]
                }

    def parse_pymupdf(self, file_path: str) -> Tuple[List[Tuple[int, str]], Dict[str, str]]:
        """
        Extracts and return the page blocks and metadata from the PDF file
        Then determines which blocks of text should be merged, depending on whether the previous block was a
        paragraph header
        """
        logger.info("Extracting pdf metadata")
        doc = fitz.open(file_path)
        # print(f"parse_pymupdf: doc.metadata = {doc.metadata}")
        metadata = self.get_metadata(file_path, doc.metadata)
        # print(f"parse_pymupdf: metadata = {metadata}")
        pages = []
        page_with_max_text = -1
        max_page_text_length = -1

        # for each page in pdf file
        logger.info("Extracting text from pdf file")
        for i, page in enumerate(doc.pages()):
            first_block_of_page = True
            prv_block_text = ""
            prv_block_is_valid = True
            prv_block_is_paragraph = False
            # obtain the blocks
            blocks = page.get_text("blocks")

            # for each block
            for block in blocks:
                # only consider text blocks
                # if block["type"] == 0:
                if block[6] == 0:
                    block_is_valid = True
                    block_is_pagenr = False
                    block_is_paragraph = False
                    # block_tag = pdf_analyzer.get_block_tag(doc_tags, i, block_id)
                    # block_text = pdf_analyzer.get_block_text(doc_tags, i, block_id)
                    block_text = block[4]

                    # block text should not represent a page header or footer
                    pattern_pagenr = r'^\s*(\d+)([.\s]*)$|^\s*(\d+)([.\s]*)$'
                    if bool(re.match(pattern_pagenr, block_text)):
                        block_is_pagenr = True
                        block_is_valid = False
                        # print(f"block {block[5]}: {block_text} is a page number")

                    # block text should not represent a page header or footer containing a pipe character
                    # and some text
                    pattern_pagenr = r'^\s*(\d+)\s*\|\s*([\w\s]+)$|^\s*([\w\s]+)\s*\|\s*(\d+)$'
                    if bool(re.match(pattern_pagenr, block_text)):
                        block_is_pagenr = True
                        block_is_valid = False
                        # print(f"block {block[5]}: {block_text} is a page number")

                    # block text should not represent any form of paragraph title
                    pattern_paragraph = r'^\d+(\.\d+)*\s*.+$'
                    if bool(re.match(pattern_paragraph, block_text)):
                        if not block_is_pagenr:
                            block_is_paragraph = True
                            # print(f"block {block[5]}: {block_text} is a paragraph")

                    # if current block is content
                    if block_is_valid and (not block_is_paragraph):
                        # print(f"block {block[5]} is valid and not a paragraph: {block_text} ")
                        # and the previous block was a paragraph
                        if prv_block_is_paragraph:
                            # extend the paragraph block text with a newline character and the current block text
                            block_text = prv_block_text + "\n" + block_text
                        # but if the previous block was a content block
                        else:
                            if prv_block_is_valid and block_is_valid:
                                # extend the content block text with a whitespace character and the current block text
                                block_text = prv_block_text + " " + block_text
                        # in both cases, set the previous block text to the current block text
                        prv_block_text = block_text
                    # else if current block text is not content
                    else:
                        # and the current block is not the very first block of the page
                        if not first_block_of_page:
                            # if previous block was content
                            if prv_block_is_valid and (not prv_block_is_paragraph):
                                # add text of previous block to pages together with page number
                                pages.append((i, prv_block_text))
                                # print(f"added to page {i}: {prv_block_text}")
                                # and empty the previous block text
                                prv_block_text = ""
                            # if previous block was not relevant
                            else:
                                # just set the set the previous block text to the current block text
                                prv_block_text = block_text

                    # set previous block validity indicators to current block validity indicators
                    prv_block_is_valid = block_is_valid
                    # prv_block_is_pagenr = block_is_pagenr
                    prv_block_is_paragraph = block_is_paragraph
                    prv_block_text = block_text

                    # set first_block_of_page to False
                    first_block_of_page = False

            # end of page:
            # if previous block was content
            if prv_block_is_valid and (not prv_block_is_paragraph):
                # add text of previous block to pages together with page number
                pages.append((i, prv_block_text))
                # print(f"added to page {i}: {prv_block_text}")

            # store pagenr with maximum amount of characters for language detection of document
            page_text_length = len(pages[i][1])
            if page_text_length > max_page_text_length:
                page_with_max_text = i
                max_page_text_length = page_text_length

            # tabs = page.find_tables() # locate and extract any tables on page
            # print(f"{len(tabs.tables)} table found on {page}") # display number of found tables
            # if tabs.tables:  # at least one table found?
            #     pprint.pprint(tabs[0].extract())  # print content of first table

        metadata['Language'] = metadata['Language'] if 'Language' in metadata.keys() else \
            ut.detect_language(pages[page_with_max_text][1])
        logger.info(f"The language detected for this document is {metadata['Language']}")

        return pages, metadata
    
    def parse_html(self, file_path: str) -> Tuple[List[Tuple[int, str]], Dict[str, str]]:
        """
        Extract and return the pages and metadata from the html file
        """
        path_to_pdf = self.convert_html_to_pdf(file_path)
        # load text and extract raw page
        pages, metadata = self.parse_pymupdf(path_to_pdf)
        
        metadata["last_change_time"] = os.stat(file_path).st_mtime

        return pages, metadata

    def parse_txt(self, file_path: str) -> Tuple[List[Tuple[int, str]], Dict[str, str]]:
        """
        Extract and return the pages and metadata from the text file
        """
        path_to_pdf = self.convert_txt_to_pdf(file_path)

        pages, metadata = self.parse_pymupdf(path_to_pdf)

        metadata["last_change_time"] = os.stat(file_path).st_mtime

        return pages, metadata

    def parse_word(self, file_path: str) -> Tuple[List[Tuple[int, str]], Dict[str, str]]:
        """
        Extract and return the pages and metadata from the word document
        """
        # convert docx to pdf
        path_to_pdf = self.convert_docx_to_pdf(file_path)
        # load text and extract raw page
        pages, metadata = self.parse_pymupdf(path_to_pdf)

        metadata["last_change_time"] = os.stat(file_path).st_mtime

        return pages, metadata

    def convert_docx_to_pdf(self, docx_path, pdf_path=None):
        folder, file = os.path.split(docx_path)
        pdf_path = os.path.join(folder,"conversions",file+'.pdf')

        if not os.path.exists(os.path.join(folder,'conversions')):
            os.mkdir(os.path.join(folder,'conversions'))

        convert(docx_path, pdf_path)
        return pdf_path
    
    def convert_txt_to_pdf(self, file_path):
        # Create a PDF object
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        folder, file = os.path.split(file_path)
        pdf_path = os.path.join(folder,"conversions",file+'.pdf')

        if not os.path.exists(os.path.join(folder,'conversions')):
            os.mkdir(os.path.join(folder,'conversions'))

        # Read the content of the text file
        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                # Remove non-ASCII characters using regex
                # otherwise we have to download fonts
                clean_line = re.sub(r'[^\x00-\x7F]+', '', line)
                pdf.multi_cell(190, 10, txt=clean_line, align="L")

        # Save the PDF to a specified location
        pdf.output(pdf_path)

        return pdf_path
    
    def convert_html_to_pdf(self, file_path):
        path_wkhtmltopdf = r"C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe"
        config = pdfkit.configuration(wkhtmltopdf=path_wkhtmltopdf)

        with open(file_path, "r", encoding="utf-8") as file:
            html = file.read()

        folder, file = os.path.split(file_path)
        pdf_path = os.path.join(folder,"conversions",file+'.pdf')

        if not os.path.exists(os.path.join(folder,'conversions')):
            os.mkdir(os.path.join(folder,'conversions'))

        try:
            pdfkit.from_string(html, pdf_path, configuration=config)
            logger.info(f"PDF created successfully at: {pdf_path}")
        except Exception as e:
            logger.info(f"Failed to create PDF: {e}")

        return pdf_path