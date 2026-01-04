import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from newsapi import NewsApiClient
from transformers import pipeline
from datetime import datetime, timedelta
from collections import Counter
import re
from typing import List, Dict
import warnings

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Market Sentiment Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f9ff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3b82f6;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'sentiment_model' not in st.session_state:
    st.session_state.sentiment_model = None
if 'articles_df' not in st.session_state:
    st.session_state.articles_df = None

# Helper functions
@st.cache_resource
def load_sentiment_model():
    """Load the sentiment analysis model (cached for performance)"""
    try:
        # Using FinBERT model which is specifically trained for financial sentiment
        # Fallback to a general sentiment model if FinBERT is not available
        model = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            max_length=512,
            truncation=True
        )
        return model, "FinBERT (Financial Sentiment)"
    except Exception:
        try:
            # Fallback to Twitter RoBERTa model (good for short texts)
            model = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                max_length=512,
                truncation=True
            )
            return model, "Twitter-RoBERTa (General Sentiment)"
        except Exception:
            # Final fallback to DistilBERT
            model = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                max_length=512,
                truncation=True
            )
            return model, "DistilBERT (General Sentiment)"

def fetch_news(api_key: str, query: str, from_date: str, to_date: str, language: str = 'en') -> List[Dict]:
    """Fetch news articles using NewsAPI"""
    try:
        newsapi = NewsApiClient(api_key=api_key)
        
        # Fetch articles
        all_articles = newsapi.get_everything(
            q=query,
            from_param=from_date,
            to=to_date,
            language=language,
            sort_by='relevancy',
            page_size=100  # Max for free tier
        )
        
        if all_articles['status'] == 'ok':
            return all_articles['articles']
        else:
            return []
    except Exception as e:
        st.error(f"Error fetching news: {str(e)}")
        return []

def analyze_sentiment(text: str, model) -> Dict:
    """Analyze sentiment of a given text"""
    try:
        if not text or len(text.strip()) == 0:
            return {'label': 'NEUTRAL', 'score': 0.5}
        
        result = model(text[:512])[0]  # Limit to 512 tokens
        
        # Normalize labels to POSITIVE, NEGATIVE, NEUTRAL
        label = result['label'].upper()
        if 'POS' in label or label == 'LABEL_2':
            normalized_label = 'POSITIVE'
        elif 'NEG' in label or label == 'LABEL_0':
            normalized_label = 'NEGATIVE'
        else:
            normalized_label = 'NEUTRAL'
        
        return {
            'label': normalized_label,
            'score': result['score']
        }
    except Exception as e:
        return {'label': 'NEUTRAL', 'score': 0.5}

def extract_keywords(texts: List[str], top_n: int = 10) -> List[tuple]:
    """Extract top keywords from texts"""
    # Simple keyword extraction using word frequency
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                  'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
                  'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                  'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those'}
    
    all_words = []
    for text in texts:
        if text:
            words = re.findall(r'\b[a-z]{3,}\b', text.lower())
            all_words.extend([w for w in words if w not in stop_words])
    
    return Counter(all_words).most_common(top_n)

def create_sentiment_plot(df: pd.DataFrame):
    """Create sentiment trend visualization"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot 1: Sentiment Score Over Time
    daily_sentiment = df.groupby('date').agg({
        'sentiment_score': 'mean',
        'title': 'count'
    }).reset_index()
    daily_sentiment.columns = ['date', 'avg_sentiment', 'article_count']
    
    ax1.plot(daily_sentiment['date'], daily_sentiment['avg_sentiment'], 
             marker='o', linewidth=2, markersize=6, color='#3b82f6')
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.fill_between(daily_sentiment['date'], daily_sentiment['avg_sentiment'], 0, 
                      where=(daily_sentiment['avg_sentiment'] > 0), alpha=0.3, color='green', label='Positive')
    ax1.fill_between(daily_sentiment['date'], daily_sentiment['avg_sentiment'], 0, 
                      where=(daily_sentiment['avg_sentiment'] <= 0), alpha=0.3, color='red', label='Negative')
    ax1.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Average Sentiment Score', fontsize=12, fontweight='bold')
    ax1.set_title('Daily Sentiment Trend', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Plot 2: Article Volume by Sentiment
    sentiment_counts = df.groupby(['date', 'sentiment_label']).size().unstack(fill_value=0)
    sentiment_counts.plot(kind='bar', stacked=True, ax=ax2, 
                          color={'POSITIVE': '#22c55e', 'NEGATIVE': '#ef4444', 'NEUTRAL': '#94a3b8'})
    ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Number of Articles', fontsize=12, fontweight='bold')
    ax2.set_title('Article Volume by Sentiment', fontsize=14, fontweight='bold')
    ax2.legend(title='Sentiment', title_fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    return fig

# Main App
st.markdown('<p class="main-header">📈 Market Sentiment & Trend Analyzer</p>', unsafe_allow_html=True)
st.markdown("### Analyze market sentiment from real-time news using AI-powered NLP")

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # API Key Input
    api_key = st.text_input(
        "NewsAPI.org API Key",
        type="password",
        help="Get your free API key at https://newsapi.org/register"
    )
    
    st.markdown("---")
    
    # Model Info
    if st.button("🔄 Load Sentiment Model"):
        with st.spinner("Loading AI model..."):
            model, model_name = load_sentiment_model()
            st.session_state.sentiment_model = model
            st.success(f"✅ Loaded: {model_name}")
    
    if st.session_state.sentiment_model:
        st.info("✓ Model Ready")
    
    st.markdown("---")
    
    # Instructions
    st.markdown("### 📖 How to Use")
    st.markdown("""
    1. Enter your NewsAPI key above
    2. Load the sentiment model
    3. Enter a company ticker or name
    4. Select date range
    5. Click 'Analyze Market'
    """)
    
    st.markdown("---")
    st.markdown("### 🔑 Free API Key")
    st.markdown("[Get NewsAPI Key →](https://newsapi.org/register)")
    st.caption("Free tier: 100 requests/day")

# Main Content
col1, col2 = st.columns([2, 1])

with col1:
    query = st.text_input(
        "🏢 Enter Company/Ticker Symbol",
        placeholder="e.g., Tesla, AAPL, Microsoft, Bitcoin",
        help="Enter company name, ticker symbol, or market keyword"
    )

with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    analyze_button = st.button("🚀 Analyze Market", type="primary", use_container_width=True)

# Date Range Selection
col3, col4 = st.columns(2)
with col3:
    from_date = st.date_input(
        "From Date",
        value=datetime.now() - timedelta(days=7),
        max_value=datetime.now()
    )
with col4:
    to_date = st.date_input(
        "To Date",
        value=datetime.now(),
        max_value=datetime.now()
    )

# Analysis Section
if analyze_button:
    if not api_key:
        st.error("⚠️ Please enter your NewsAPI key in the sidebar!")
    elif not query:
        st.warning("⚠️ Please enter a company or ticker symbol!")
    elif not st.session_state.sentiment_model:
        st.error("⚠️ Please load the sentiment model first (click button in sidebar)!")
    else:
        with st.spinner(f"🔍 Fetching news for '{query}'..."):
            articles = fetch_news(
                api_key=api_key,
                query=query,
                from_date=from_date.strftime('%Y-%m-%d'),
                to_date=to_date.strftime('%Y-%m-%d')
            )
        
        if not articles:
            st.warning("📭 No articles found for the given search criteria. Try adjusting your query or date range.")
        else:
            st.success(f"✅ Found {len(articles)} articles. Analyzing sentiment...")
            
            # Progress bar for sentiment analysis
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Process articles
            processed_articles = []
            for idx, article in enumerate(articles):
                # Combine title and description for better analysis
                text = f"{article.get('title', '')} {article.get('description', '')}"
                
                # Analyze sentiment
                sentiment = analyze_sentiment(text, st.session_state.sentiment_model)
                
                # Calculate sentiment score (-1 to 1)
                if sentiment['label'] == 'POSITIVE':
                    score = sentiment['score']
                elif sentiment['label'] == 'NEGATIVE':
                    score = -sentiment['score']
                else:
                    score = 0
                
                processed_articles.append({
                    'title': article.get('title', 'N/A'),
                    'description': article.get('description', 'N/A'),
                    'source': article.get('source', {}).get('name', 'Unknown'),
                    'published_at': article.get('publishedAt', 'N/A'),
                    'url': article.get('url', '#'),
                    'sentiment_label': sentiment['label'],
                    'sentiment_confidence': sentiment['score'],
                    'sentiment_score': score
                })
                
                # Update progress
                progress = (idx + 1) / len(articles)
                progress_bar.progress(progress)
                status_text.text(f"Analyzing article {idx + 1}/{len(articles)}")
            
            progress_bar.empty()
            status_text.empty()
            
            # Create DataFrame
            df = pd.DataFrame(processed_articles)
            df['published_at'] = pd.to_datetime(df['published_at'])
            df['date'] = df['published_at'].dt.date
            st.session_state.articles_df = df
            
            # Display Metrics
            st.markdown("---")
            st.subheader("📊 Sentiment Overview")
            
            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            
            with metric_col1:
                avg_sentiment = df['sentiment_score'].mean()
                sentiment_emoji = "🟢" if avg_sentiment > 0.1 else "🔴" if avg_sentiment < -0.1 else "🟡"
                st.metric("Overall Sentiment", f"{sentiment_emoji} {avg_sentiment:.3f}", 
                         delta=None)
            
            with metric_col2:
                positive_pct = (df['sentiment_label'] == 'POSITIVE').sum() / len(df) * 100
                st.metric("Positive Articles", f"{positive_pct:.1f}%", 
                         delta=f"{(df['sentiment_label'] == 'POSITIVE').sum()} articles")
            
            with metric_col3:
                negative_pct = (df['sentiment_label'] == 'NEGATIVE').sum() / len(df) * 100
                st.metric("Negative Articles", f"{negative_pct:.1f}%",
                         delta=f"{(df['sentiment_label'] == 'NEGATIVE').sum()} articles")
            
            with metric_col4:
                neutral_pct = (df['sentiment_label'] == 'NEUTRAL').sum() / len(df) * 100
                st.metric("Neutral Articles", f"{neutral_pct:.1f}%",
                         delta=f"{(df['sentiment_label'] == 'NEUTRAL').sum()} articles")
            
            # Sentiment Visualization
            st.markdown("---")
            st.subheader("📈 Sentiment Trends")
            
            fig = create_sentiment_plot(df)
            st.pyplot(fig)
            
            # Trending Keywords
            st.markdown("---")
            st.subheader("🔥 Trending Keywords")
            
            all_texts = (df['title'] + ' ' + df['description']).tolist()
            keywords = extract_keywords(all_texts, top_n=15)
            
            if keywords:
                col_kw1, col_kw2 = st.columns(2)
                
                with col_kw1:
                    # Display as tags
                    keyword_html = " ".join([
                        f'<span style="background-color: #3b82f6; color: white; padding: 5px 10px; '
                        f'margin: 3px; border-radius: 15px; display: inline-block; font-size: 14px;">'
                        f'{word} ({count})</span>'
                        for word, count in keywords[:8]
                    ])
                    st.markdown(keyword_html, unsafe_allow_html=True)
                
                with col_kw2:
                    keyword_html = " ".join([
                        f'<span style="background-color: #3b82f6; color: white; padding: 5px 10px; '
                        f'margin: 3px; border-radius: 15px; display: inline-block; font-size: 14px;">'
                        f'{word} ({count})</span>'
                        for word, count in keywords[8:]
                    ])
                    st.markdown(keyword_html, unsafe_allow_html=True)
            
            # Recent Articles
            st.markdown("---")
            st.subheader("📰 Recent Articles")
            
            # Filter options
            filter_col1, filter_col2 = st.columns(2)
            with filter_col1:
                sentiment_filter = st.multiselect(
                    "Filter by Sentiment",
                    options=['POSITIVE', 'NEGATIVE', 'NEUTRAL'],
                    default=['POSITIVE', 'NEGATIVE', 'NEUTRAL']
                )
            with filter_col2:
                num_articles = st.slider("Number of articles to display", 5, 50, 10)
            
            filtered_df = df[df['sentiment_label'].isin(sentiment_filter)].sort_values('published_at', ascending=False).head(num_articles)
            
            # Display articles
            for _, row in filtered_df.iterrows():
                sentiment_color = {
                    'POSITIVE': '#22c55e',
                    'NEGATIVE': '#ef4444',
                    'NEUTRAL': '#94a3b8'
                }[row['sentiment_label']]
                
                with st.expander(f"**{row['title']}** - {row['source']} ({row['published_at'].strftime('%Y-%m-%d %H:%M')})"):
                    st.markdown(
                        f'<div style="border-left: 4px solid {sentiment_color}; padding-left: 10px;">'
                        f'<p><strong>Sentiment:</strong> <span style="color: {sentiment_color}; font-weight: bold;">'
                        f'{row["sentiment_label"]}</span> (Confidence: {row["sentiment_confidence"]:.2%})</p>'
                        f'<p>{row["description"]}</p>'
                        f'<p><a href="{row["url"]}" target="_blank">Read full article →</a></p>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
            
            # Download Data
            st.markdown("---")
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Analysis Data (CSV)",
                data=csv,
                file_name=f"sentiment_analysis_{query}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray; padding: 20px;'>"
    "Built with ❤️ using Streamlit, NewsAPI, Hugging Face Transformers & Matplotlib<br>"
    "<small>Data powered by NewsAPI.org | AI powered by Hugging Face</small>"
    "</div>",
    unsafe_allow_html=True
)
