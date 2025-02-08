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

import bs4
import os
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

class WebDocuments:

    def __init__(self, web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",)):
        '''Create Document for RAG system from Web Pages.

        Parameters:
        web_paths (tuple): A tuple of web paths to load the documents from.
        '''

        bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))
        loader = WebBaseLoader(
            web_paths=web_paths,
            bs_kwargs={"parse_only": bs4_strainer},
        )

        self.docs = loader.load()
        
    def _split(self, chunk_size=1000, chunk_overlap=200):
        '''Split the documents into chunks.

        Parameters:
        chunk_size (int): The size of the chunk.
        chunk_overlap (int): The overlap between the chunks.
        '''
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True
            )
        
        all_splits = text_splitter.split_documents(self.docs)

        return all_splits
    
    def get_vecstore(self,
                 chunk_size=1000,
                 chunk_overlap=200,
                 embedder="OpenAI"):
        '''Create a vector store from the documents.

        Parameters:
        chunk_size (int): The size of the chunk.
        chunk_overlap (int): The overlap between the chunks.
        embedder (str): The embedder to use. Default is OpenAI.
        '''
        all_splits = self._split(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        vectorstore = Chroma.from_documents(documents=all_splits, embedding=self._get_embedder(embedder))

        return vectorstore
    
    def _get_embedder(self, embedder):
        '''Get the embedder for the documents.'''
        if embedder == "OpenAI":
            embd = OpenAIEmbeddings()
        else:
            model_name = "sentence-transformers/all-mpnet-base-v2"
            model_kwargs = {"device": "cpu"}
            try:
                embd = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)
            except Exception as ex:
                print("Exception: ", ex)
                local_model_path = "/kaggle/input/sentence-transformers/minilm-l6-v2/all-MiniLM-L6-v2"
                print(f"Use alternative (local) model: {local_model_path}\n")
                embd = HuggingFaceEmbeddings(model_name=local_model_path, model_kwargs=model_kwargs)
        return embd