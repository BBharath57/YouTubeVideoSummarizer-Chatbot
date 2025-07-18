# YouTubeVideoSummarizer-Chatbot

YouTube Chatbot with RAG (Retrieval-Augmented Generation)
This Jupyter Notebook implements a backend chatbot that can answer questions based on the transcript of a given YouTube video. It leverages Retrieval-Augmented Generation (RAG) principles to fetch relevant information from the video transcript and generate informed responses using a Large Language Model (LLM).

Features
YouTube Transcript Extraction: Automatically fetches transcripts from YouTube videos using the youtube_transcript_api.

Text Chunking: Splits the lengthy video transcript into smaller, manageable chunks for efficient processing.

Vectorization: Converts text chunks into numerical vector embeddings using GoogleGenerativeAIEmbeddings for semantic search.

Vector Store (FAISS): Stores and indexes the generated embeddings using FAISS for fast similarity search (retrieval).

Retrieval-Augmented Generation (RAG) Chain: Combines the retrieval of relevant text chunks with an LLM (gemini-2.0-flash) to generate context-aware answers.

Conversational Interface (Console): Provides a simple console-based interface for interacting with the chatbot.

Dependencies
To run this notebook, you need the following Python libraries. You can install them using pip:

pip install youtube_transcript_api langchain langchain_core langchain_community langchain_huggingface langchain_google_genai faiss-cpu

Setup and Usage
Google API Key:

Obtain a Google API Key from the Google Cloud Console (ensure the Generative Language API is enabled).

Replace the placeholder google_api_key='' in the notebook with your actual API key:

embed = GoogleGenerativeAIEmbeddings(model='models/embedding-001', google_api_key='YOUR_API_KEY_HERE')
# and
model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key='YOUR_API_KEY_HERE', temperature=0.5)

Note: For production or sharing, consider using environment variables or a secure secrets management system instead of hardcoding your API key.

Run the Notebook:

Open the Youtube-Chatbot-RAG.ipynb file in a Jupyter environment (e.g., Jupyter Lab, VS Code with Jupyter extension, Google Colab).

Run all cells sequentially.

Provide YouTube URL:

In the "Youtube Transcript" section, update the link variable with the URL of the YouTube video you want to analyze.

The notebook will then extract, chunk, and vectorize the transcript.

Interact with the Chatbot:

The final section of the notebook contains a while True loop that initiates the console-based chatbot.

Type your questions at the You :  prompt.

Type exit to end the conversation.

Example interaction:

You :  explain in detail about agent and give some examples
AI :  Hello!

AI agents are intelligent systems that receive a high-level goal from a user and autonomously plan, decide, and execute a sequence of actions using external tools, APIs, or knowledge sources. They maintain context, reason over multiple steps, and adapt to new information.
...

Core Components Explained
youtube_transcript_api: Used to programmatically fetch transcripts from YouTube videos.

RecursiveCharacterTextSplitter: A utility from LangChain to break down large texts into smaller, overlapping chunks, which is crucial for RAG to fit content into LLM context windows.

GoogleGenerativeAIEmbeddings: Generates dense vector representations (embeddings) of text chunks, allowing for semantic similarity searches.

FAISS: A library for efficient similarity search and clustering of dense vectors. It's used here as the vector store to index and retrieve relevant text chunks quickly.

PromptTemplate: Defines the structure of the prompt sent to the LLM, incorporating the retrieved context and the user's question.

ChatGoogleGenerativeAI: The LangChain integration for Google's Gemini models (specifically gemini-2.0-flash), used as the main LLM for generating responses.

LangChain Expression Language (LCEL) Components (RunnableParallel, RunnablePassthrough, RunnableLambda, StrOutputParser): These are used to construct the RAG chain efficiently, defining the flow of data from retrieval to prompt formatting to model invocation and output parsing.

Disclaimer
This notebook demonstrates the backend logic of a RAG chatbot. For a web-based user interface, you would typically integrate this backend with a frontend framework like Streamlit, Flask, or React.
