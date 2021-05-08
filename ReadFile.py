import fitz

# This code is read Text data from PDFs using Fitz(aka PyMuPDF)

class ReadPDF():
    def __init__(self, file):
        self.file = file 

    def ReturnPDFText(self):
        doc =  fitz.open(stream=self.file.read(), filetype="pdf")
        text = ""
        for page in doc:
            text += page.getText().strip()
        return text