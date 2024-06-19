import argparse
# from dataclasses import dataclass
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Access the API key
openai_api_key = os.getenv('OPENAI_API_KEY')

CHROMA_PATH = "chroma"

INTENT_PROMPT_TEMPLATES = {
    "personal_growth": """
    The user is seeking guidance on personal growth. Provide relevant information, advice, or practices based on spiritual texts, teachings, and general spiritual wisdom. Ensure the response addresses the user's concern in a compassionate and insightful manner.
    """,
    "managing_emotions": """
    The user is seeking guidance on managing emotions. Provide relevant information, advice, or practices based on spiritual texts, teachings, and general spiritual wisdom. Ensure the response addresses the user's concern in a compassionate and insightful manner.
    """,
    "spiritual_practices": """
    The user is seeking guidance on spiritual practices. Provide relevant information, advice, or practices based on spiritual texts, teachings, and general spiritual wisdom. Ensure the response addresses the user's interest in a compassionate and insightful manner.
    """,
    # Add more intents as needed
}

# Default prompt template for generic queries
DEFAULT_PROMPT_TEMPLATE = """
Given the user's query about {query_text}, provide relevant information, guidance, or advice based on spiritual texts, teachings, and general spiritual wisdom like Atma Sutra, Karma Sutra, Gita, etc. Ensure the response is tailored to address the user's concern or question in a compassionate and insightful manner.
"""

def answer_query(query_text, user_intent=None):
    try:
        # Prepare the DB.
        embedding_function = OpenAIEmbeddings(openai_api_key=openai_api_key)
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

        # Search the DB.
        results = db.similarity_search_with_relevance_scores(query_text, k=5)
        if len(results) == 0 or results[0][1] < 0.7:
            # Generate response using OpenAI chatbot directly
            model = ChatOpenAI()
            response_text = model.predict(query_text)

            formatted_response = f"Response: {response_text}\nSources: No sources available (query passed directly to chatbot)"
            return {
                'response': response_text,
                'sources': None  # No sources available when query is passed directly to chatbot
            }

        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        
        # Select the appropriate prompt template
        if user_intent and user_intent in INTENT_PROMPT_TEMPLATES:
            prompt_template_str = INTENT_PROMPT_TEMPLATES[user_intent]
        else:
            prompt_template_str = DEFAULT_PROMPT_TEMPLATE
        
        # Format the prompt with context and query
        prompt_template = ChatPromptTemplate.from_template(prompt_template_str)
        prompt = prompt_template.format(context=context_text, query_text=query_text)

        model = ChatOpenAI()
        response_text = model.predict(prompt)

        sources = [doc.metadata.get("source", None) for doc, _score in results]
        formatted_response = f"Response: {response_text}\nSources: {sources}"
        return {
            'response': response_text,
            'sources': sources
        }
    except Exception as e:
        return {
            'response': 'An error occurred while processing your request.',
            'sources': None
        }


if __name__ == "__main__":
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    parser.add_argument("--intent", type=str, help="The user's intent.", default=None)
    args = parser.parse_args()
    query_text = args.query_text
    user_intent = args.intent
    response = answer_query(query_text, user_intent)