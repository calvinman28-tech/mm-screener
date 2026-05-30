import datetime
import os
import requests
from bs4 import BeautifulSoup
from flask import Flask, Response, request, render_template, render_template_string
import yfinance as yf
from functools import wraps

app = Flask(__name__)

# A sample watchlist of US stocks. Expand this list as needed.
WATCHLIST = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "NVDA",
    "META",
    "TSLA",
    "AMD",
    "NFLX",
    "COST",
]

_MARKET_CAP_CACHE = {}
_UNIVERSE_CACHE = None

def fetch_wikipedia_table(url, table_index=0):
    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        tables = soup.find_all("table", {"class": "wikitable"})
        if table_index < len(tables):
            return tables[table_index]
    except Exception as exc:
        print(f"Unable to fetch Wikipedia table from {url}: {exc}")
    return None


def parse_symbols_from_wikitable(table, symbol_column_names):
    if table is None:
        return []

    headers = [th.get_text(strip=True) for th in table.find_all("th")]
    symbol_col = None
    for idx, header in enumerate(headers):
        normalized = header.lower()
        if any(name.lower() in normalized for name in symbol_column_names):
            symbol_col = idx
            break
    if symbol_col is None:
        return []

    symbols = []
    for row in table.find_all("tr")[1:]:
        cells = row.find_all("td")
        if len(cells) > symbol_col:
            symbol = cells[symbol_col].get_text(strip=True)
            if symbol:
                symbols.append(symbol.replace(".", "-"))
    return symbols


def get_sp500_tickers():
    table = fetch_wikipedia_table(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", table_index=0
    )
    return parse_symbols_from_wikitable(table, ["Symbol"])


def get_nasdaq_largecap_tickers():
    table = fetch_wikipedia_table("https://en.wikipedia.org/wiki/Nasdaq-100", table_index=4)
    return parse_symbols_from_wikitable(table, ["Ticker"])


def get_market_cap(ticker):
    if ticker in _MARKET_CAP_CACHE:
        return _MARKET_CAP_CACHE[ticker]

    market_cap = None
    try:
        yf_ticker = yf.Ticker(ticker)
        info = getattr(yf_ticker, "info", None) or {}
        market_cap = info.get("marketCap")
        if market_cap is None:
            fast_info = getattr(yf_ticker, "fast_info", None)
            if fast_info is not None:
                market_cap = fast_info.get("market_cap")
    except Exception as exc:
        print(f"Error fetching market cap for {ticker}: {exc}")

    if isinstance(market_cap, (int, float)):
        _MARKET_CAP_CACHE[ticker] = market_cap
    else:
        market_cap = None

    return market_cap


def get_us_large_cap_tickers(min_market_cap=3_000_000_000):
    global _UNIVERSE_CACHE
    if _UNIVERSE_CACHE is None:
        sp500 = get_sp500_tickers()
        nasdaq = get_nasdaq_largecap_tickers()
        combined = sorted({ticker.replace(".", "-") for ticker in sp500 + nasdaq})
        _UNIVERSE_CACHE = combined

    # S&P 500 and Nasdaq-100 constituents are already large-cap names.
    # We return the full list of tickers and rely on a safer screen subset below.
    return _UNIVERSE_CACHE


def check_auth(username, password):
    user = os.getenv("FLASK_USER", "admin")
    pw = os.getenv("FLASK_PASS", "MMscreener")
    return username == user and password == pw


def authenticate():
    return Response(
        "Authentication required", 401,
        {"WWW-Authenticate": "Basic realm='Login Required'"},
    )


def requires_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.authorization
        if not auth or not check_auth(auth.username, auth.password):
            return authenticate()
        return f(*args, **kwargs)
    return decorated


def screen_minervini_stocks(tickers):
    screened_stocks = []

    # Request roughly 1.5 years of data to comfortably calculate 200-day SMA and lookback trends
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=550)

    for ticker in tickers:
        try:
            # Fetch historical data
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, end=end_date)

            if len(df) < 200:
                continue  # Skip if there isn't enough historical data

            # 1. Calculate technical indicators
            df["SMA_50"] = df["Close"].rolling(window=50).mean()
            df["SMA_150"] = df["Close"].rolling(window=150).mean()
            df["SMA_200"] = df["Close"].rolling(window=200).mean()

            # Get 52-week high and low from historical window
            # (Approx. 252 trading days in a year)
            df_52w = df.iloc[-252:]
            low_52w = df_52w["Close"].min()
            high_52w = df_52w["Close"].max()

            # Latest values
            current_price = df["Close"].iloc[-1]
            sma_50 = df["SMA_50"].iloc[-1]
            sma_150 = df["SMA_150"].iloc[-1]
            sma_200 = df["SMA_200"].iloc[-1]

            # 200-day SMA value from 20 trading days ago (~1 month)
            sma_200_20d_ago = df["SMA_200"].iloc[-20]

            # 2. Evaluate Mark Minervini's Template Criteria
            cond_1 = (current_price > sma_150) and (current_price > sma_200)
            cond_2 = sma_150 > sma_200
            cond_3 = sma_200 > sma_200_20d_ago  # 200 SMA trending up for 1 month
            cond_4 = (sma_50 > sma_150) and (sma_50 > sma_200)
            cond_5 = current_price > sma_50
            cond_6 = (
                current_price >= high_52w * 0.75
            )  # Within 25% of 52-week high
            cond_7 = (
                current_price >= low_52w * 1.30
            )  # At least 30% above 52-week low

            # If all criteria match, append to the final list
            if (
                cond_1
                and cond_2
                and cond_3
                and cond_4
                and cond_5
                and cond_6
                and cond_7
            ):
                screened_stocks.append(
                    {
                        "Ticker": ticker,
                        "Price": round(current_price, 2),
                        "SMA_50": round(sma_50, 2),
                        "SMA_150": round(sma_150, 2),
                        "SMA_200": round(sma_200, 2),
                        "52W_Low": round(low_52w, 2),
                        "52W_High": round(high_52w, 2),
                    }
                )
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            continue

    return screened_stocks


@app.route("/")
@requires_auth
def index():
    universe = get_us_large_cap_tickers(min_market_cap=3_000_000_000)
    # Screen a larger subset to include more tickers while still keeping the page responsive.
    subset = universe[:600]
    results = screen_minervini_stocks(subset)
    return render_template("index.html", stocks=results)


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=int(os.getenv("PORT", 5000)))