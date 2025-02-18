# Financial Insights API with FastAPI & Ngrok

This project provides an AI-powered financial insights API using an open-source LLM (LLaMA, Falcon, GPT-J, Mistral). The API generates financial health summaries, risk assessments, and personalized recommendations. The project is deployed using FastAPI and exposed publicly via Ngrok.

## Features
- **LLM-based Financial Insights**: Uses an open-source model to generate financial summaries.
- **Risk Assessments**: Classifies financial risk levels (High, Medium, Low).
- **Personalized Recommendations**: Provides actionable financial advice.
- **FastAPI Backend**: Serves AI-generated responses through a REST API.
- **Ngrok Integration**: Exposes the FastAPI service publicly for easy access.

## Prerequisites
Ensure you have the following installed:
- Python 3.8+
- pip & virtual environment tools
- FastAPI
- Uvicorn
- Transformers (Hugging Face)
- Bitsandbytes (for model quantization)
- Ngrok (for public access)

## Installation
```bash
# Clone the repository
git clone https://github.com/your-repo/financial-insights-api.git
cd financial-insights-api

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Setting Up Ngrok
```bash
# Install ngrok (if not installed)
pip install pyngrok

# Authenticate Ngrok (replace with your token)
ngrok authtoken YOUR_NGROK_AUTH_TOKEN

# Expose FastAPI on port 8000
ngrok http 8000
```
After running Ngrok, note the **public URL** it provides and use it to test the API.

## API Endpoints
### 1. Generate Financial Insights
**POST /generate_insight**
```json
{
  "transaction_history": "string",
  "credit_score": 0,
  "debt_to_income": 0
}

```
### 2. Health Check
**GET /health**
```json
{
  "status": "ok"
}
```

## Deployment
For production, you can deploy using Docker, Azure, or any cloud platform.

## Troubleshooting
- **CUDA Error in Bitsandbytes**: Ensure you have a CUDA-compatible GPU.
- **Ngrok Authentication Failed**: Sign up at https://dashboard.ngrok.com/ and set up your token.

## Contributing
Feel free to open an issue or submit a pull request to improve the project.

## License
MIT License

