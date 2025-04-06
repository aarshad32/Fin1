import re
import os
import sys

# Initialize availability flags
NUMPY_AVAILABLE = False
SKLEARN_AVAILABLE = False
SPACY_AVAILABLE = False

# Try to import numpy
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    print("NumPy not available. Using basic fallback methods for NLP.")

# Try to import scikit-learn
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    print("scikit-learn not available. Using basic fallback methods for NLP.")

# Try to import and load spaCy
try:
    import spacy
    
    # Try different paths for the model
    try:
        # Try standard loading
        nlp = spacy.load('en_core_web_sm')
        SPACY_AVAILABLE = True
        print("Successfully loaded spaCy model")
    except OSError:
        # Try downloading if not found
        print("Attempting to download spaCy model...")
        os.system(f"{sys.executable} -m spacy download en_core_web_sm")
        try:
            nlp = spacy.load('en_core_web_sm')
            SPACY_AVAILABLE = True
            print("Successfully downloaded and loaded spaCy model")
        except Exception as e:
            print(f"Error loading spaCy model after download: {str(e)}")
except ImportError:
    print("spaCy not available. Using basic fallback methods for NLP.")

print("NLP Processor module loaded with available modules:") 
print(f"NumPy: {NUMPY_AVAILABLE}, scikit-learn: {SKLEARN_AVAILABLE}, spaCy: {SPACY_AVAILABLE}")

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
    """Preprocess text without relying on NLTK tokenization"""
    # Convert to lowercase
    text = text.lower()
    
    # Manual tokenization to avoid nltk dependencies
    # Split by spaces and remove punctuation
    tokens = []
    for word in text.split():
        # Remove trailing punctuation
        clean_word = word.strip('.,;:!?()[]{}""\'')
        if clean_word:
            tokens.append(clean_word)
    
    # Filter out common stop words
    stop_words = {'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
                 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
                 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
                 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
                 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
                 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
                 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
                 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
                 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
                 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
                 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
                 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now'}
    
    filtered_tokens = [word for word in tokens if word not in stop_words and len(word) > 1]
    
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

def identify_query_type(query, company):
    """Identify the type of financial query"""
    # Preprocess the query
    preprocessed_query = preprocess_text(query)
    
    # Extract financial terms from the query
    financial_terms = extract_financial_terms(query)
    
    # Default best match based on financial terms (fallback method)
    term_based_match = None
    if financial_terms:
        # Use the first category found as a fallback
        term_based_match = next(iter(financial_terms))
    
    # If sklearn is not available, just use the term-based match
    if not SKLEARN_AVAILABLE:
        print("Using basic term matching for query identification")
        best_match = term_based_match
        confidence = 0.5 if best_match else 0.3  # Default confidence
        
        # If no match from terms, try to find a match based on keywords
        if best_match is None:
            query_lower = query.lower()
            
            # Check for common keywords to assign a query type
            if any(word in query_lower for word in ['revenue', 'sales', 'earn']):
                best_match = 'revenue_query'
            elif any(word in query_lower for word in ['profit', 'income', 'earnings']):
                best_match = 'net_income_query'
            elif any(word in query_lower for word in ['asset', 'debt', 'liability']):
                best_match = 'assets_liabilities_query'
            elif any(word in query_lower for word in ['cash', 'flow', 'liquidity']):
                best_match = 'cash_flow_query'
            elif any(word in query_lower for word in ['growth', 'growing', 'increase']):
                best_match = 'growth_query'
            elif any(word in query_lower for word in ['performance', 'overview', 'summary']):
                best_match = 'performance_query'
            elif any(word in query_lower for word in ['compare', 'comparison', 'versus', 'vs']):
                best_match = 'comparison_query'
            elif any(word in query_lower for word in ['trend', 'pattern', 'historical']):
                best_match = 'trend_query'
            elif any(word in query_lower for word in ['forecast', 'future', 'predict', 'projection']):
                best_match = 'forecast_query'
            else:
                # If still no match, default to performance query
                best_match = 'performance_query'
                confidence = 0.2
    else:
        # Use TF-IDF and cosine similarity for more accurate matching if sklearn is available
        # Prepare the query templates with the company name
        formatted_templates = {}
        for query_type, templates in QUERY_TEMPLATES.items():
            formatted_templates[query_type] = [preprocess_text(template.format(company=company)) for template in templates]
        
        # Calculate similarity scores between the query and templates
        best_match = None
        highest_score = -1
        
        for query_type, templates in formatted_templates.items():
            try:
                # Create a TF-IDF vectorizer (already checked SKLEARN_AVAILABLE)
                from sklearn.feature_extraction.text import TfidfVectorizer
                from sklearn.metrics.pairwise import cosine_similarity
                
                vectorizer = TfidfVectorizer()
                
                # Combine the preprocessed query with the templates for this query type
                all_texts = [preprocessed_query] + templates
                
                # Transform the texts to TF-IDF vectors
                tfidf_matrix = vectorizer.fit_transform(all_texts)
                
                # Calculate cosine similarity between the query and each template
                for i in range(1, len(all_texts)):
                    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[i:i+1])[0][0]
                    if similarity > highest_score:
                        highest_score = similarity
                        best_match = query_type
            except Exception as e:
                print(f"Error in TF-IDF processing for {query_type}: {str(e)}")
                continue
        
        # If TF-IDF method failed, fall back to term-based match
        if best_match is None:
            best_match = term_based_match
            
        # Determine confidence based on score
        confidence = highest_score if highest_score > 0 else 0.3
    
    # If we still have no match, default to performance query
    if best_match is None:
        best_match = 'performance_query'
        confidence = 0.2
    
    return {
        'query_type': best_match,
        'confidence': confidence,
        'financial_terms': financial_terms
    }

def analyze_query(query, company):
    """Analyze a financial query and extract meaningful information"""
    # Get the query type and extracted information
    query_info = identify_query_type(query, company)
    
    # Check if the query is about comparison
    is_comparison = 'comparison' in query_info.get('financial_terms', {})
    
    # Check if the query is about trends
    is_trend = 'trend' in query_info.get('financial_terms', {})
    
    # Check if the query is about forecasting
    is_forecast = 'forecast' in query_info.get('financial_terms', {})
    
    # Extract time periods if mentioned
    time_periods = []
    year_pattern = r'\b(20\d{2})\b'
    years = re.findall(year_pattern, query)
    if years:
        time_periods = years
    
    # Determine if specific metrics are requested
    metrics = []
    financial_terms = query_info.get('financial_terms', {})
    
    if 'revenue' in financial_terms:
        metrics.append('revenue')
    if 'net_income' in financial_terms:
        metrics.append('net_income')
    if 'assets' in financial_terms:
        metrics.append('assets')
    if 'liabilities' in financial_terms:
        metrics.append('liabilities')
    if 'cash_flow' in financial_terms:
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
        'financial_terms': query_info.get('financial_terms', {})
    }
