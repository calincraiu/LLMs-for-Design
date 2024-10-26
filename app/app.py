import dotenv

from contextlib import asynccontextmanager
from fastapi import FastAPI, Response
from apputil import *

# Constants
MODEL_OUTPUT_SCHEMA_FILE = "../resources/model_output_schema/facade_specification_output_schema.json"
"""The json file that describes the structure of the model output - this will be parsed to create a dynamic pydantic BaseModel."""
DOCUMENT_DIRECTORY = "../resources/documents"
"""The directory containing all the documents to be used for RAG. Note that only PDFs are supported."""
CHUNK_SIZE = 6000
"""The size in number of characters for each chunk - this represents a segment of the data loaded from the PDF documents."""
CHUNK_OVERLAP = 800
"""The size in number of characters for the overlap between adjacent chunks. 0 would mean there is no overlap."""
MODEL_NAME = 'models/gemini-1.5-flash'
"""The name of the Gemini model that will be used in the chain. Available options: 'models/gemini-1.5-flash', 'models/gemini-1.5-pro', 'models/gemini-1.0-pro'"""
LLM_TEMPERATURE = 0.3
"""Set the temperature of the model. For Gemini models, this should be a float between 0 and 2."""

# Runtime
runtime = {
    "llm_qa_chain": None, # The chain used to query the llm using RAG and a structured prompt.
    "parser":  None, # The parser used to enforce a structured model output.
    "prompt_template": None # The template that will be used to merge the user query, structured output instructions and retrieved document chunk data into a single prompt.
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup events
    print("Starting application.")

    # Load environment variables
    loaded: bool = dotenv.load_dotenv(dotenv_path="../.env")
    print("Loaded environment variables." if loaded else "Could not load environment variables.")
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
    genai.configure(api_key=GOOGLE_API_KEY)
    print("Loaded the Google API key.")

    with open(MODEL_OUTPUT_SCHEMA_FILE) as f:
        try:
            model_output_json_schema = json.load(f)
        except:
            raise Exception("Could not load json from the specified file.")

    # Load all documents from the given directory and chunk them
    text_chunks = get_document_chunks(DOCUMENT_DIRECTORY, CHUNK_SIZE, CHUNK_OVERLAP)

    # Create a vectorstore from the text chunks, using the set model to embed them
    vectorstore_retriever = create_vectorstore_retriever(GOOGLE_API_KEY, text_chunks)
    
    # Load the specified Gemini model
    print("Configuring model...", end="")
    llm = ChatGoogleGenerativeAI(model=MODEL_NAME, google_api_key=GOOGLE_API_KEY, temperature=LLM_TEMPERATURE)
    print("Done.")

    # Create a chain for prompting, retrieving chunks from the vectorstore and using them in the LLM to produce an answer
    runtime["llm_qa_chain"] = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore_retriever, return_source_documents=True)

    # Create model fields to define a structured LLM output from JSON specs
    model_fields = get_model_fields_from_json(model_output_json_schema)
    # Dynamically create a Pydantic model - class that represents the structured output of the LLM
    DynamicModel = create_model('DynamicModel', **model_fields)

    # Create a parser that will process the model outputs into the predefined pydantic class
    runtime["parser"] = PydanticOutputParser(pydantic_object=DynamicModel)

    # Create a prompt template to query the model
    runtime["prompt_template"] = ChatPromptTemplate(
        messages=[
            HumanMessagePromptTemplate.from_template(
                "Answer the user's question as best as possible with information related to the fields specified in the output structure." + \
                "\n{format_instructions}\n{question}"
            )
        ],
        input_variables=["question"],
        partial_variables={
            "format_instructions": runtime["parser"].get_format_instructions(),
        }
    )

    yield
    # Shutdown events
    print("Shutting application down.")
    runtime.clear()

# Instantiate the FastAPI app with the given startup/shutdown lifespan instructions
app = FastAPI(lifespan=lifespan)

@app.get("/")
def read_root():
    return {"Description": "This app is a demo of using LLMs and RAG for generating structured outputs in the context of archtiectural design."}

@app.post("/generate")
def generate(query: str):
    # Use the prompt template and the user query to create the input to the model
    input = runtime["prompt_template"].format_prompt(question=query)
    # Invoke the LLM chain with a string representation of the input prompt
    output = runtime["llm_qa_chain"].invoke(input.to_string())
    # Parse the output into the predefined pydantic class
    parsed_output: BaseModel = runtime["parser"].parse(output["result"])
    # Return the structured model response as json
    return Response(content=parsed_output.model_dump_json(), media_type="application/json")