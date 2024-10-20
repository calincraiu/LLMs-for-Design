import argparse
import json
import os
import os.path
from typing import Optional

import dotenv
import PIL.Image
from pydantic import BaseModel, create_model, Field
from typing import Any

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
    """Check if a given string is valid json."""
    try:
        json.loads(json_string)
    except ValueError as e:
        return False
    return True

def get_model_fields_from_json(json: dict) -> dict:
    """Create fields for a dynamic pydantic class that will act as the structured LLM output."""
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


def main():
    # Define available models
    available_models: list = ['models/gemini-1.5-flash', 'models/gemini-1.5-pro', 'models/gemini-1.0-pro']
    available_models_display_string: str = " ".join(available_models)

    # Parse input arguments
    parser = argparse.ArgumentParser(description='Process some inputs.')
    parser.add_argument('--query', type=str, required=True, help='Query that will be included in the prompt to the LLM.')
    parser.add_argument('--model', type=str, help='Name of the model to use in the LLM application. Select an available Gemini model.', choices=available_models, default='models/gemini-1.5-flash')
    parser.add_argument('--temperature', type=float, help='Set the temperature of the model. For Gemini models, this should be a float between 0 and 2.', default=0.3)
    parser.add_argument('--document_for_RAG', type=str, required=True, help='The path to a PDF file that will be used as a source of data for generating outputs with the LLM using RAG.')
    parser.add_argument('--chunk_size', type=int, help='The size, in number of characters, of the chunks to break the loaded document into.', default=5000)
    parser.add_argument('--chunk_overlap', type=int, help='The size, in number of characters, of the overlap between the chunks that the document is broken into.', default=800)
    parser.add_argument('--model_output_spec_json_string', type=str, help='JSON string defining the structure of the LLM output. This is used to define a pydantic BaseModel class.', default=None)
    parser.add_argument('--model_output_spec_json_path', type=str, help='path to JSON file defining the structure of the LLM output. This is used to define a pydantic BaseModel class.', default=None)

    args = parser.parse_args()

    # Process inputs
    query: str = args.query
    if len(query) == 0:
        raise Exception("The provided query can't be null or empty. Please provide a non-empty string.")

    model_name: str = args.model
    if model_name not in available_models:
        raise Exception(f"The specified model must be one of the available options: {available_models_display_string}")
    
    temperature: float = args.temperature
    if temperature < 0.0 or temperature > 2.0:
        raise Exception("The specified llm temperature must be a number between 0 and 2.")
    
    document_path: str = args.document_for_RAG
    if not os.path.isfile(document_path):
        raise Exception("The specified document filepath is not a valid path. Check if the file exists.")
    
    chunk_size: int = args.chunk_size
    if chunk_size <= 0:
        raise Exception("The specified chunk size must be an integer larger than 0.")
    
    chunk_overlap: int = args.chunk_overlap
    if chunk_overlap < 0:
        raise Exception("The specified chunk overlap must be an integer larger or equal to 0.")
    
    model_output_spec_json_string = args.model_output_spec_json_string
    model_output_spec_json_path = args.model_output_spec_json_path

    if model_output_spec_json_string != None and model_output_spec_json_path != None:
        raise Exception("You must specify the model output JSON schema by either providing the json string OR the path to a json file. " + \
            "You may not specify both.")
    
    if model_output_spec_json_string == None and model_output_spec_json_path == None:
        raise Exception("You must specify the model output JSON schema by either providing the json string OR the path to a json file. " + \
            "You did not specify either.")

    model_output_json_schema = None
    if model_output_spec_json_string != None:
        if not is_valid_json(model_output_spec_json_string):
            raise Exception("The provided string is not valid JSON.")
        else:
            model_output_json_schema = json.loads(model_output_spec_json_string)
    elif model_output_spec_json_path != None:
        if not os.path.isfile(model_output_spec_json_path):
            raise Exception("The specified model output spec filepath is not a valid path. Check if the file exists.")
        else:
            with open(model_output_spec_json_path) as f:
                try:
                    model_output_json_schema = json.load(f)
                except:
                    raise Exception("The provided model output schema loaded from the given file is not valid JSON.")

    # Load environment variables
    loaded: bool = dotenv.load_dotenv(dotenv_path="../.env")
    print("Loaded environment variables." if loaded else "Could not load environment variables.")
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    genai.configure(api_key=GEMINI_API_KEY)
    print("Loaded GEMINI API KEY.")

    # Load the specified PDF document as a data source
    document_loader = PyPDFLoader(document_path)
    document_pages = document_loader.load_and_split()
    print(f"Loaded the specified document for RAG: {len(document_pages)} pages")

    # Create chunks from the loaded PDF pages
    character_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    text_chunks = character_splitter.split_text("\n\n".join(str(page.page_content) for page in document_pages))
    print(f"Created {len(text_chunks)} chunks of size {chunk_size} from the original {len(document_pages)} document pages.")

    # Set the model that will embed the text chunks
    embeddings_model = "models/embedding-001"
    embeddings = GoogleGenerativeAIEmbeddings(model=embeddings_model, google_api_key=GEMINI_API_KEY)

    # Create a vectorstore from the text chunks, using the set model to embed them
    number_of_chunks_to_retrieve_per_query = 5
    vectorstore_retriever = Chroma.from_texts(texts=text_chunks, embedding=embeddings).as_retriever(search_kwargs={"k":number_of_chunks_to_retrieve_per_query})
    
    # Load the specified Gemini model
    print("Configuring model...", end="")
    llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=GEMINI_API_KEY, temperature=temperature)
    print("Done.")

    # Create a chain for prompting, retrieving chunks from the vectorstore and using them in the LLM to produce an answer
    llm_qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore_retriever, return_source_documents=True)

    # Create model fields to define a structured LLM output from JSON specs
    model_fields = get_model_fields_from_json(model_output_json_schema)
    # Dynamically create a Pydantic model - class that represents the structured output of the LLM
    DynamicModel = create_model('DynamicModel', **model_fields)

    # Create a parser that will process the model outputs into the predefined pydantic class
    parser = PydanticOutputParser(pydantic_object=DynamicModel)

    # Create a prompt template to query the model
    prompt_template = ChatPromptTemplate(
        messages=[
            HumanMessagePromptTemplate.from_template(
                "Answer the user's question as best as possible with information about the cladding system, glazing system, materials and glazing percentage in the context of the discussed facade design." + \
                "\n{format_instructions}\n{question}"
            )
        ],
        input_variables=["question"],
        partial_variables={
            "format_instructions": parser.get_format_instructions(),
        }
    )

    # Use the prompt template and the user query to create the input to the model
    input = prompt_template.format_prompt(question=query)

    # Invoke the LLM chain with a string representation of the input prompt
    output = llm_qa_chain.invoke(input.to_string())

    # parse the output into the predefined Pydantic class
    parsed_output = parser.parse(output["result"])
    print(parsed_output)


if __name__ == "__main__":
    main()