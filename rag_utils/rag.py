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

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableLambda
from langchain import hub
from langchain_core.prompts import PromptTemplate
import json

from rag_utils.documents import *
from rag_utils.generator import *
from rag_utils.retriever import *
from time import time

class RAG:

    def __init__(
        self,
        retriever,
        generator,
        prompt_src="rlm/rag-prompt",
        cache_dir="",
    ):
        # the retriever and generator
        self.retriever = retriever
        self.generator = generator

        # the rag chain
        if (prompt_src == "custom"):
            self.prompt = self._custom_prompt()
        else:
            self.prompt = hub.pull(prompt_src)
        self.rag_chain = self._get_chain()

        # the rag chain trace
        self.template = None
        self.trace_retrieved_docs = None
        self.trace_prompted_docs = None

        # the cache directory (where to store generated files)
        self.cache_dir = cache_dir

    # ----------------------------------------------------------------------------
    # rag chain helper functions
    # ----------------------------------------------------------------------------

    def _format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    def _custom_prompt(self):
        '''Create a custom RAG prompt.'''

        self.template = \
            "User:\n" + \
            "Use the following pieces of context to answer the question at the end." + \
            "If you don't know the answer, just say that you don't know, don't try to make up an answer." + \
            "Use three sentences maximum and keep the answer as concise as possible." + \
            "Always say \"thanks for asking!\" at the end of the answer.\n" + \
            "Context:\n{context}\n" + \
            "Question:\n{question}\n" + \
            "Answer:"

        custom_rag_prompt = PromptTemplate.from_template(self.template)
        return custom_rag_prompt

    def _get_chain(self):
        '''Get the RAG chain.'''
        
        rag_chain = (
            {
                "context": RunnableLambda(self.retriever.retrieve) | self._trace_retrieved_docs | self._format_docs,
                "question": RunnablePassthrough()
            }
            | self.prompt | self._trace_prompted_docs
            | RunnableLambda(self.generator.gen_resp)
            | StrOutputParser()
        )
        return rag_chain
    
    # ----------------------------------------------------------------------------
    # chain trace functions
    # ----------------------------------------------------------------------------

    def _trace_retrieved_docs(self, docs):
        '''Trace the retrieved documents.
        
        Parameters:
        docs (list): The list of retrieved documents.
        '''
        self.trace_retrieved_docs = [{"page_content":doc.page_content, "metadata": doc.metadata} for doc in docs]
        return docs
    
    def _trace_prompted_docs(self, message):
        '''Trace the prompted documents.
        
        Parameters:
        message (str): The message to prompt the documents.
        '''
        self.trace_prompted_docs = message.to_string()
        return message

    # ----------------------------------------------------------------------------
    # response functions
    # ----------------------------------------------------------------------------
    
    def gen_resp_dict(self, query):
        '''Generate a response to a query.
        
        Parameters:
        query (str): The query to generate a response to.
        '''

        time_start = time()  # Start the timer

        response = self.rag_chain.invoke(query)

        time_end = time()  # End the timer
        total_time = f"{round(time_end-time_start, 3)} sec"  # Calculate the total time

        return {"prompt template": self.template,
                "docs": self.trace_retrieved_docs,
                "input": self.trace_prompted_docs,
                "response": response,
                "time": total_time}
    
    def save_resp(self, resp_dict):
        '''Save the response to a file.
        
        Parameters:
        resp_dict (dict): The response dictionary.
        '''

        os.makedirs(self.cache_dir, exist_ok=True)

        with open(f"{self.cache_dir}/response.json", "w") as f:
            json.dump(resp_dict, f, indent=4)