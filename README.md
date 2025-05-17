# Fraud Suraksha

An AI-powered fraud detection assistant that helps users verify if messages, persons, or situations are potentially fraudulent.

## 🎥 Demo
https://github-production-user-asset-6210df.s3.amazonaws.com/118119552/444748979-99b99e13-bddc-43c2-a1e8-eef7cd6bdf71.mp4?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20250517%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20250517T054304Z&X-Amz-Expires=300&X-Amz-Signature=1f208c1f95c97cf924cefa79e479d774734f975ae0e5021db774987b2d9f2dab&X-Amz-SignedHeaders=host

## Features

- RAG-based fraud analysis using Google's Gemini model
- Conversation memory to maintain context
- Detailed fraud risk assessment
- Recommendations for user actions
- Simple, intuitive chat interface

## Prerequisites

1. Python 3.8 or higher installed on your system
2. Git for cloning the repository
3. Windows operating system (for run.bat)

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/dhanushperumalla/Fraud-Suraksha
   cd fraud-suraksha
   ```

2. Create a `.env` file in the project root with your Google AI API key:
   ```
   GOOGLE_API_KEY=your_google_api_key_here
   ```

3. Run the startup utility:
   ```bash
   run.bat
   ```

   The utility will:
   - Create a virtual environment if it doesn't exist
   - Install required dependencies
   - Provide options to launch the app or reset the database

## Running the App

Using the startup utility (run.bat), you have three options:

1. **Launch Fraud Suraksha app** - Starts the main application
2. **Reset database** - Fixes connection issues by resetting the vector database
3. **Exit** - Closes the utility

The app will be available at http://localhost:8501 in your web browser.

## Troubleshooting

If you encounter database connection errors:

1. Choose option 2 "Reset database" from the run.bat menu
2. Wait for the reset to complete
3. Return to the menu and choose option 1 to launch the app

## Usage

1. Type your question or describe a suspicious situation in the chat input
2. The AI will analyze the information against known fraud patterns
3. Review the AI's assessment and recommendations
4. Continue the conversation for more detailed analysis

## Example Use Cases

- Verifying if a Google Maps related message is legitimate
- Checking if a request for reviews/ratings is a scam
- Assessing potential phishing attempts
- Evaluating suspicious business practices
