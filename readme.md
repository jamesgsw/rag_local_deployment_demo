# Local RAG deployment POC
I have created this repository to create a proof of concpet (POC) project on development of Large Language Model(LLM) with Retrieval Augmented Generation (RAG) approach.

## Structure
* I have tested locally where the LLM and embeddings models are ran with models that are loaded locally on my machine under the file name main_local.py
* I have another version where the LLM and embedding models are hosted on OpenAI endpoints and allows anyone to run the model through a Docker container if you have the api_key set up in an .env file

## Set-up
* Create a .env file in the parent folder
* Add the OPENAI_API_KEY environment variable
(This will allow you to connect to the LLM and embedding endpoint)
* cd into the project root directory
  * Run docker build -t rag-app .
  * docker run --rm -it rag-app
