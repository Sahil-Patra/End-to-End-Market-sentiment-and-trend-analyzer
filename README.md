
# 📈 End-to-End Market Sentiment & Trend Analyzer

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-ff4b4b)
![AI Model](https://img.shields.io/badge/Model-FinBERT-yellow)
![NLP](https://img.shields.io/badge/NLP-TF--IDF%20%26%20Transformers-green)

## 📖 Overview
The **Market Sentiment Analyzer** is an enterprise-ready AI web application designed to help traders and financial analysts gauge market sentiment in real-time. By combining **Natural Language Processing (NLP)**, **Hugging Face Transformers**, and the **NewsAPI**, the app fetches financial news, runs high-performance batch sentiment inference, and surface actionable insights.

## 🚀 Key Features
- **Real-Time Financial News Stream:** Integrates with NewsAPI to retrieve news for any stock ticker, market keyword, or company (e.g., AAPL, NVDA, Tesla, Bitcoin).
- **High-Performance Batch Inference:** Optimized Hugging Face pipeline using vectorized batching (`batch_size=16`) for up to **10x faster sentiment analysis**.
- **Financial NLP Models:** Uses `ProsusAI/finbert` (a BERT model fine-tuned on financial text) with automatic fallback to DistilBERT.
- **TF-IDF Keyword & N-Gram Extraction:** Advanced natural language feature extraction (`scikit-learn`) to filter out noise and highlight key market buzzwords.
- **Trend & Volume Visualization:** 
  - Time-series sentiment trajectory graphs.
  - Categorical article distribution charts with memory-safe rendering.
- **Seamless Resource Caching:** Uses `@st.cache_resource` for automatic background model initialization.
- **Data Export:** Instant CSV export for offline research and backtesting.

## 🛠️ Tech Stack
- **Frontend / UX:** Streamlit
- **Data & Feature Engineering:** Pandas, NumPy, Scikit-Learn (`TfidfVectorizer`)
- **Machine Learning & NLP:** Hugging Face Transformers, PyTorch (`FinBERT`)
- **Visualization:** Matplotlib, Seaborn
- **API Integration:** NewsAPI.org

## ⚙️ Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Sahil-Patra/End-to-End-Market-sentiment-and-trend-analyzer.git
   cd market-sentiment-analyzer
   ```

2. **Create a virtual environment (Optional but Recommended)**
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Mac/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Get an API Key**
   - Sign up for a free key at [NewsAPI.org](https://newsapi.org/).

5. **Run the Application**
   ```bash
   streamlit run app.py
   ```

## 📊 How It Works
1. **Input:** Enter a target Stock Ticker or Company Name (e.g., "NVIDIA").
2. **Configuration:** Select the date range and load the AI model.
3. **Processing:** The app fetches news metadata and runs it through the Transformer model.
4. **Result:** View the "Sentiment Score," "Article Volume," and "Trending Keywords" on the dashboard.

## 🤖 Models Used
The application dynamically selects the best available model for the task:
1. **Primary:** `ProsusAI/finbert` (Financial Sentiment)
2. **Fallback:** `cardiffnlp/twitter-roberta-base-sentiment` (Social/General)
3. **Fastest:** `distilbert-base-uncased`

## 📁 Project Structure
```
├── app.py                  # Main application logic
├── requirements.txt        # Python dependencies
├── .gitignore             # Files to exclude from Git
├── README.md               # Project documentation
└── assets/                 # Images for README
```

## 🔮 Future Improvements
- [ ] Add real-time stock price overlay on sentiment graphs.
- [ ] Implement email alerts for significant sentiment shifts.
- [ ] Add support for multiple stock comparisons side-by-side.

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## 📜 License
This project is open-source and available under the MIT License.


---

