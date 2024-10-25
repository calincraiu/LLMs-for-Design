import json
import os
import os.path
from typing import Optional

from pydantic import BaseModel, create_model, Field
from typing import Any, List

import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate


def is_valid_json(json_string: str) -> bool:
    """Check if a given string is valid json.
    
    Args:
        json_string: JSON given as a string.

    Returns:
       Whether or not the given string represents valid JSON.
    """
    try:
        json.loads(json_string)
    except ValueError as e:
        return False
    return True


def get_model_fields_from_json(json: dict) -> dict:
    """Create fields for a dynamic pydantic class that will act as the structured LLM output.
    
    Args:
        json: JSON given as a dictionary.

    Returns:
        A dictionary of objects created from the json specifications - that represent the fields of a pydantic BaseModel.
    """
    # Extract the fields definition from the JSON
    field_definitions = json.get("fields", {})
    # Process the model fields
    model_fields = {}
    for field_name, field_info in field_definitions.items():
        field_type = field_info.get("type", str)
        is_optional = field_info.get("optional", False)
        description = field_info.get("description", "")
        # Handling Optional fields
        if is_optional:
            field_type = Optional[field_type]
        # Add the field to the model, using Field() for default and description
        model_fields[field_name] = (field_type, Field(description=description))
    return model_fields


def get_pdf_paths(directory_path):
  """Gets all PDF file paths within a given directory.

  Args:
    directory_path: The path to the directory to search.

  Returns:
    A list of PDF file paths.
  """

  pdf_paths = []
  for root, dirs, files in os.walk(directory_path):
    for file in files:
      if file.endswith(".pdf"):
        pdf_paths.append(os.path.join(root, file))
  return pdf_paths


def get_document_chunks(directory_path: str, chunk_size:int = 6000, chunk_overlap: int = 800) -> list[str]:
    """Given the path to a directory, load all PDF files in it and create text chunks as a list of strings.
    
    Args:
    directory_path: The path to the directory that contains the documents for chunking.
    chunk_size: The size in number of characters for each chunk.
    chunk_size: The size in number of characters for the overlap between adjacent chunks.

  Returns:
    A list of strings. Each string is a separate chunk.
  """
    chunks = []
    loaded_documents = 0
    loaded_pages = 0

    pdf_filepaths = get_pdf_paths(directory_path)

    for filepath in pdf_filepaths:
        # Load and split each file
        document_loader = PyPDFLoader(filepath)
        document_pages = document_loader.load_and_split()
        # Create chunks from the loaded PDF pages
        character_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        text_chunks = character_splitter.split_text("\n\n".join(str(page.page_content) for page in document_pages))
        chunks.extend(text_chunks)
        loaded_documents += 1
        loaded_pages += len(document_pages)

    print(f"Created {len(chunks)} chunks of size {chunk_size} from the original {loaded_pages} document pages of {loaded_documents} documents.")
    return chunks


def create_vectorstore_retriever(google_api_key: str, text_chunks: list[str], embedding_model: str = "models/embedding-001", chunks_to_retrieve_per_query: int = 5) -> Chroma:
    """Create a Chroma vectorstore retriever using a GoogleGenerativeAIEmbeddings embedding model.
    
    Args:
    google_api_key: The Google API key used to access the embedding model.
    text_chunks: The text chunks to embed.
    embedding_model: The name of the embedding model.
    chunks_to_retrieve_per_query: The number of relevant chunks to be retrieved from the vectorstore when it is queried.

  Returns:
    A Chroma vectorstore as a retriever.
  """
    
   # Set the model that will embed the text chunks
    embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model, google_api_key=google_api_key)

    # Create a vectorstore from the text chunks, using the set model to embed them
    vectorstore_retriever = Chroma.from_texts(texts=text_chunks, embedding=embeddings).as_retriever(search_kwargs={"k":chunks_to_retrieve_per_query})
    return vectorstore_retriever