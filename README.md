# RAG Chatbot Instructions

## Setup
1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Configure API Key**:
    - Open `.env` file.
    - Replace `your_api_key_here` with your actual OpenAI API Key.

## Running the App
Run the following command in your terminal:
```bash
streamlit run app.py
```

## Usage
1.  **Upload File**: Use the sidebar to upload a PDF or TXT file.
2.  **Chat**: Ask questions in the chat input box.
3.  **Performance**: After an answer is generated, expand the "Show RAG Performance" section to see latency and source documents.
4.  **System Prompt**: You can modify `system_prompt.txt` to change the bot's behavior.
"# Traditional_RAG" 
