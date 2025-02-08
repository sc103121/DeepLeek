# ----------------------------------------------------------------------------
# NOTICE: This code is the exclusive property of Cornell University
#         Computer Architecture Research and is strictly confidential.
#
#         Unauthorized distribution, reproduction, or use of this code, in
#         whole or in part, is strictly prohibited. This includes, but is
#         not limited to, any form of public or private distribution,
#         publication, or replication.
#
# For inquiries or access requests, please contact:
#         Zuoming Fu (zf242@cornell.edu)
# ----------------------------------------------------------------------------


from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader

class Retriever:
    
    def __init__(self,
                 vectorstore: Chroma,
                 search_type="similarity",
                 search_kwargs={"k": 6}):
        '''A retriever that uses a vectorstore to retrieve documents

        Parameters:
        vectorstore (Chroma): The vectorstore to use for retrieval.
        search_type (str): The type of search to use.
        search_kwargs (dict): The keyword arguments to pass to the search function.
        '''

        self.vectorstore = vectorstore
        self.retriever = vectorstore.as_retriever(search_type=search_type, search_kwargs=search_kwargs)

    def retrieve(self, query):
        '''Retrieve documents based on a query.

        Parameters:
        query (str): The query to retrieve documents for.
        '''
        results = self.retriever.invoke(query)
        return results