# Fraud Suraksha

An AI-powered fraud detection assistant that helps users verify if messages, persons, or situations are potentially fraudulent.

## Features

- RAG-based fraud analysis using Google's Gemini model
- Conversation memory to maintain context
- Detailed fraud risk assessment
- Recommendations for user actions
- Simple, intuitive chat interface

## Setup

1. Make sure you have Python 3.8+ installed
2. Install required packages:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the project root with your Google AI API key:
   ```
   GOOGLE_API_KEY=your_google_api_key_here
   ```
4. Ensure `GMaps.pdf` is in the project directory (this is pre-loaded with Google Maps scam information)

## Running the App

To start the Fraud Suraksha assistant:

```
streamlit run app.py
```

The app will be available at http://localhost:8501 in your web browser.

## Usage

1. Type your question or describe a suspicious situation in the chat input
2. The AI will analyze the information against known fraud patterns
3. Review the AI's assessment and recommendations
4. Continue the conversation for more detailed analysis

## Troubleshooting

If you encounter the error "Could not connect to tenant default_tenant" or other database-related issues:

1. Run the reset script to clear the database:
   ```
   python reset_db.py
   ```
2. Restart the application:
   ```
   streamlit run app.py
   ```

This will recreate the vector database from scratch and should resolve most connection issues.

## Example Use Cases

- Verifying if a Google Maps related message is legitimate
- Checking if a request for reviews/ratings is a scam
- Assessing potential phishing attempts
- Evaluating suspicious business practices 