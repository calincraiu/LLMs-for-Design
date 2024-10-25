import argparse
import dotenv
from apputil import *


def main():
    # Define available models
    available_models: list = ['models/gemini-1.5-flash', 'models/gemini-1.5-pro', 'models/gemini-1.0-pro']
    available_models_display_string: str = " ".join(available_models)

    # Parse input arguments
    parser = argparse.ArgumentParser(description='Process some inputs.')
    parser.add_argument('--query', type=str, required=True, help='Query that will be included in the prompt to the LLM.')
    parser.add_argument('--model', type=str, help='Name of the model to use in the LLM application. Select an available Gemini model.', choices=available_models, default='models/gemini-1.5-flash')
    parser.add_argument('--temperature', type=float, help='Set the temperature of the model. For Gemini models, this should be a float between 0 and 2.', default=0.3)
    parser.add_argument('--document_directory', type=str, required=True, help='The path to a directory containing PDF files - that will be used as a source of data for generating outputs with the LLM using RAG.')
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
    
    document_directory: str = args.document_directory
    if not os.path.isdir(document_directory):
        raise Exception("The specified document directory path is not a valid path. Check if the directory exists.")
    
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
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
    genai.configure(api_key=GOOGLE_API_KEY)
    print("Loaded the Google API key.")

    # Load all documents from the given directory and chunk them
    text_chunks = get_document_chunks(document_directory, chunk_size, chunk_overlap)

    # Create a vectorstore from the text chunks, using the set model to embed them
    vectorstore_retriever = create_vectorstore_retriever(GOOGLE_API_KEY, text_chunks)
    
    # Load the specified Gemini model
    print("Configuring model...", end="")
    llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=GOOGLE_API_KEY, temperature=temperature)
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
                "Answer the user's question as best as possible with information related to the fields specified in the output structure." + \
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