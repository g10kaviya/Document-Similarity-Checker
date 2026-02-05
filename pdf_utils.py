""" 
PDF text extraction utility.
This function reads all pages and concatenates their extracted text.
"""

import PyPDF2

def extract_text(pdf_path):
    """Return concatenated text for all pages in the PDF located at pdf_path.

    Args:
        pdf_path (str): path to a PDF file

    Returns:
        str: concatenated page texts (empty string if no text found)
    """
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text
