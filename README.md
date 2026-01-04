
# 📈 End-to-End Market Sentiment & Trend Analyzer

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-ff4b4b)
![AI Model](https://img.shields.io/badge/Model-FinBERT-yellow)

## 📖 Overview
The **Market Sentiment Analyzer** is an AI-powered web application designed to help traders and investors gauge market sentiment in real-time. By leveraging **Natural Language Processing (NLP)** and the **NewsAPI**, the app fetches the latest financial news, analyzes the sentiment (Positive/Negative/Neutral) using the **FinBERT** model, and visualizes trends to identify emerging market opportunities.

## 🚀 Key Features
- **Real-Time News Fetching:** Integrates with NewsAPI to pull the latest articles for any stock ticker or company (e.g., AAPL, Tesla, Bitcoin).
- **Advanced NLP Analysis:** Uses `ProsusAI/finbert` (a BERT model fine-tuned for finance) for high-accuracy sentiment detection, with automatic fallbacks to RoBERTa or DistilBERT.
- **Trend Visualization:**
  - Interactive time-series graphs showing sentiment fluctuation.
  - Stacked bar charts for article volume analysis.
- **Keyword Extraction:** Identifies trending topics and buzzwords within the news coverage.
- **Data Export:** Allows users to download the full analysis as a CSV file for further research.

## 🛠️ Tech Stack
- **Frontend:** Streamlit (Python-based web framework)
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Hugging Face Transformers, PyTorch
- **Visualization:** Matplotlib, Seaborn
- **API:** NewsAPI.org

## ⚙️ Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/market-sentiment-analyzer.git
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
└── assets/                 # (Optional) Images for README
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

