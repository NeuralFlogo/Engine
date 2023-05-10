from os import listdir
from os.path import join

from flogo.preprocessing.data_types.document import Document


class DocumentReader:

    def read(self, path):
        return {document_name: Document(join(path, document_name)) for document_name in listdir(join(path))}
