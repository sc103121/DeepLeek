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


from rag_utils import documents
from rag_utils import retriever
from rag_utils import generator
from rag_utils import rag

from langchain_openai import OpenAIEmbeddings


def main():
    '''Main function to run the RAG system.'''

    # Create the documents
    docs = documents.WebDocuments(
        web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",)
    )

    # Split the documents into chunks and create a vector store
    vecstore = docs.get_vecstore(
        chunk_size=1000,
        chunk_overlap=200,
        embedder=OpenAIEmbeddings()
    )

    # Create the retriever
    retr = retriever.Retriever(
        vectorstore=vecstore,
        search_type="similarity",
        search_kwargs={"k": 6}
    )

    # Create the generator
    gen = generator.OpenAIGenerator()

    # Create the RAG system and get the chain
    rag_system = rag.RAG(retr, gen)
    resp = rag_system.gen_resp_dict("What is an agent?")
    resp = resp["response"]

    print(resp)
    

if __name__ == "__main__":
    main()