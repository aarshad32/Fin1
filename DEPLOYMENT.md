# Deploying Your Financial Analysis Chatbot

I've prepared the necessary files to deploy your Streamlit application. While the app has been set up for Vercel deployment, there are important considerations:

## Deployment Options

### Option 1: Vercel (Limited Functionality)

The current setup includes:
- `api/index.py`: A serverless function for Vercel
- `vercel.json`: Configuration file for routing

**Important Note**: Vercel is designed for serverless functions rather than long-running processes like Streamlit applications. The current implementation will display a message indicating this limitation, but won't run the full Streamlit app.

To deploy on Vercel:
1. Push your code to GitHub
2. Connect your GitHub repository to Vercel
3. Vercel will automatically detect the configuration

### Option 2: Streamlit Cloud (Recommended)

This is the simplest and most effective option for Streamlit applications:

1. Sign up at [streamlit.io/cloud](https://streamlit.io/cloud)
2. Connect your GitHub repository
3. Select your repo, branch, and the main file (`app.py`)
4. Click "Deploy"

Your app will be instantly deployed with full functionality and you'll get a shareable URL.

### Option 3: Alternative Platforms

You can also deploy to platforms that support long-running processes:

#### Railway
1. Sign up for a Railway account at [railway.app](https://railway.app/)
2. Create a new project and link your GitHub repository
3. Set the build command to: `pip install -r requirements.txt`
4. Set the start command to: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`

#### Render
1. Sign up for a Render account at [render.com](https://render.com/)
2. Create a new Web Service and connect your GitHub repository
3. Select "Python" as the runtime
4. Set the build command to: `pip install -r requirements.txt`
5. Set the start command to: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`

## Requirements

The application requires the following packages, already specified in requirements.txt:
- matplotlib
- nltk
- numpy
- pandas
- plotly
- scikit-learn
- streamlit
- wordcloud
- yfinance
- spacy

## Getting Help

If you encounter any issues during deployment, refer to the documentation for the platform you're using:
- [Streamlit Deployment Documentation](https://docs.streamlit.io/streamlit-cloud/get-started)
- [Vercel Python Documentation](https://vercel.com/docs/functions/serverless-functions/runtimes/python)
- [Railway Documentation](https://docs.railway.app/)
- [Render Documentation](https://render.com/docs)