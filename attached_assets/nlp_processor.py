import spacy
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# Download necessary NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

# Initialize NLTK resources
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Financial domain terms to recognize in queries
FINANCIAL_TERMS = {
    'revenue': ['revenue', 'sales', 'income', 'earnings', 'money', 'earn', 'top line', 'turnover'],
    'net_income': ['net income', 'profit', 'earnings', 'bottom line', 'net earnings', 'income'],
    'assets': ['assets', 'property', 'holdings', 'possessions', 'resources'],
    'liabilities': ['liabilities', 'debt', 'obligations', 'payables'],
    'cash_flow': ['cash flow', 'cash', 'liquidity', 'funds flow', 'operating cash'],
    'growth': ['growth', 'increase', 'rise', 'improvement', 'expansion'],
    'performance': ['performance', 'results', 'achievements', 'accomplishments', 'records'],
    'comparison': ['compare', 'comparison', 'versus', 'vs', 'against', 'relative to'],
    'trend': ['trend', 'pattern', 'direction', 'movement', 'trajectory', 'history'],
    'forecast': ['forecast', 'predict', 'projection', 'outlook', 'future', 'expectation']
}

# Query templates for classification
QUERY_TEMPLATES = {
    'revenue_query': [
        "What is the revenue of {company}?",
        "Show me {company}'s revenue",
        "Tell me about {company}'s revenue",
        "What's the total revenue for {company}?",
        "How much revenue did {company} generate?",
        "What are the earnings of {company}?"
    ],
    'net_income_query': [
        "What is the net income of {company}?",
        "How has {company}'s net income changed?",
        "Tell me about {company}'s profit",
        "What's the profit for {company}?",
        "How much did {company} earn?",
        "What is {company}'s bottom line?"
    ],
    'assets_liabilities_query': [
        "What are the assets and liabilities of {company}?",
        "Tell me about {company}'s assets",
        "What is the debt of {company}?",
        "Show me {company}'s financial position",
        "What's the balance sheet of {company}?",
        "How much does {company} own and owe?"
    ],
    'cash_flow_query': [
        "How has the cash flow of {company} changed?",
        "Tell me about {company}'s cash flow",
        "What is the operating cash flow of {company}?",
        "Show me {company}'s cash situation",
        "What's the cash position for {company}?",
        "How much cash is {company} generating from operations?"
    ],
    'growth_query': [
        "What is the revenue growth of {company}?",
        "How fast is {company} growing?",
        "Tell me about {company}'s growth",
        "What's the growth rate for {company}?",
        "Is {company} growing year over year?",
        "How has {company}'s size changed over time?"
    ],
    'performance_query': [
        "How is {company} performing?",
        "Tell me about {company}'s performance",
        "What's the overall financial performance of {company}?",
        "How did {company} do financially?",
        "Give me an overview of {company}'s results",
        "What are {company}'s key financial metrics?"
    ],
    'comparison_query': [
        "Compare {company} with other companies",
        "How does {company} compare to its peers?",
        "Is {company} performing better than others?",
        "Show me a comparison of {company} with its competitors",
        "What's the relative performance of {company}?",
        "Is {company} outperforming or underperforming?"
    ],
    'trend_query': [
        "What is the trend in {company}'s revenue?",
        "Show me the trend in {company}'s performance",
        "Has {company}'s profit been increasing or decreasing?",
        "What pattern do you see in {company}'s financials?",
        "Is there a consistent trend in {company}'s metrics?",
        "How have {company}'s financials evolved over time?"
    ],
    'forecast_query': [
        "What's the forecast for {company}?",
        "Predict {company}'s future performance",
        "What can we expect from {company} in the future?",
        "Show me projections for {company}",
        "What's the outlook for {company}?",
        "How might {company} perform going forward?"
    ]
}

def preprocess_text(text):
    """Preprocess text by tokenizing, removing stopwords, and lemmatizing"""
    # Convert to lowercase
    text = text.lower()
    
    # Manual tokenization to avoid punkt_tab dependency
    # Split by spaces and remove punctuation
    tokens = []
    for word in text.split():
        # Remove trailing punctuation
        clean_word = word.strip('.,;:!?()[]{}""\'')
        if clean_word:
            tokens.append(clean_word)
    
    # Remove stop words and punctuation, then lemmatize
    filtered_tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalnum() and word not in stop_words]
    
    return " ".join(filtered_tokens)

def extract_financial_terms(text):
    """Extract financial terms from the query"""
    text = text.lower()
    extracted_terms = {}
    
    for category, terms in FINANCIAL_TERMS.items():
        for term in terms:
            if re.search(r'\b' + re.escape(term) + r'\b', text):
                if category not in extracted_terms:
                    extracted_terms[category] = []
                extracted_terms[category].append(term)
    
    return extracted_terms

def extract_entities(text):
    """Extract entities from text using spaCy"""
    doc = nlp(text)
    entities = {}
    
    for ent in doc.ents:
        if ent.label_ not in entities:
            entities[ent.label_] = []
        entities[ent.label_].append(ent.text)
    
    return entities

def identify_query_type(query, company):
    """Identify the type of financial query"""
    # Preprocess the query
    preprocessed_query = preprocess_text(query)
    
    # Prepare the query templates with the company name
    formatted_templates = {}
    for query_type, templates in QUERY_TEMPLATES.items():
        formatted_templates[query_type] = [preprocess_text(template.format(company=company)) for template in templates]
    
    # Calculate similarity scores between the query and templates
    best_match = None
    highest_score = -1
    
    for query_type, templates in formatted_templates.items():
        # Create a TF-IDF vectorizer
        vectorizer = TfidfVectorizer()
        
        # Combine the preprocessed query with the templates for this query type
        all_texts = [preprocessed_query] + templates
        
        try:
            # Transform the texts to TF-IDF vectors
            tfidf_matrix = vectorizer.fit_transform(all_texts)
            
            # Calculate cosine similarity between the query and each template
            for i in range(1, len(all_texts)):
                similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[i:i+1])[0][0]
                if similarity > highest_score:
                    highest_score = similarity
                    best_match = query_type
        except:
            # Handle case where vectorizer fails (e.g., empty texts)
            continue
    
    # Extract financial terms as a backup
    financial_terms = extract_financial_terms(query)
    
    if best_match is None and financial_terms:
        # If no template match but we found financial terms, use the first category
        best_match = next(iter(financial_terms))
    
    # Determine confidence based on score
    confidence = highest_score if highest_score > 0 else 0.3  # Default confidence if no match
    
    return {
        'query_type': best_match,
        'confidence': confidence,
        'financial_terms': financial_terms,
        'entities': extract_entities(query)
    }

def analyze_query(query, company):
    """Analyze a financial query and extract meaningful information"""
    # Get the query type and extracted information
    query_info = identify_query_type(query, company)
    
    # Check if the query is about comparison
    is_comparison = 'comparison' in query_info['financial_terms']
    
    # Check if the query is about trends
    is_trend = 'trend' in query_info['financial_terms']
    
    # Check if the query is about forecasting
    is_forecast = 'forecast' in query_info['financial_terms']
    
    # Extract time periods if mentioned
    time_periods = []
    year_pattern = r'\b(20\d{2})\b'
    years = re.findall(year_pattern, query)
    if years:
        time_periods = years
    
    # Determine if specific metrics are requested
    metrics = []
    if 'revenue' in query_info['financial_terms']:
        metrics.append('revenue')
    if 'net_income' in query_info['financial_terms']:
        metrics.append('net_income')
    if 'assets' in query_info['financial_terms']:
        metrics.append('assets')
    if 'liabilities' in query_info['financial_terms']:
        metrics.append('liabilities')
    if 'cash_flow' in query_info['financial_terms']:
        metrics.append('cash_flow')
    
    # Return the analysis results
    return {
        'query_type': query_info['query_type'],
        'confidence': query_info['confidence'],
        'is_comparison': is_comparison,
        'is_trend': is_trend,
        'is_forecast': is_forecast,
        'time_periods': time_periods,
        'metrics': metrics if metrics else ['all'],
        'financial_terms': query_info['financial_terms'],
        'entities': query_info['entities']
    }