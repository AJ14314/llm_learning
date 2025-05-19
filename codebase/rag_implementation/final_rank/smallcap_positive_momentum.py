import os
import re
from pathlib import Path

# Define the path to the smallcap stocks directory
smallcap_dir = Path(r"e:\Practise_Code\clean\stock_analysis\reports_v2\smallcap_250")

# Function to extract performance data from stock files
def extract_stock_data(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

            # Extract company name and ticker
            company_match = re.search(r'# (.*?) \((.*?)\)', content)
            if company_match:
                company_name = company_match.group(1)
                ticker = company_match.group(2)
            else:
                company_name = "Unknown"
                ticker = file_path.stem

            # Extract current price
            price_match = re.search(r'Current Price: Rs\.([\d.]+)', content)
            current_price = float(price_match.group(1)) if price_match else None

            # Extract 1-Year Change
            one_year_match = re.search(r'1-Year Change: ([\d.-]+)%', content)
            one_year_change = float(one_year_match.group(1)) if one_year_match else None

            # Extract 6-Month Change
            six_month_match = re.search(r'6-Month Change: ([\d.-]+)%', content)
            six_month_change = float(six_month_match.group(1)) if six_month_match else None

            # Extract 1-Year Momentum Rank
            one_year_rank_match = re.search(r'1-Year Momentum Rank: (\d+) out of', content)
            one_year_rank = int(one_year_rank_match.group(1)) if one_year_rank_match else None

            # Extract 6-Month Momentum Rank
            six_month_rank_match = re.search(r'6-Month Momentum Rank: (\d+) out of', content)
            six_month_rank = int(six_month_rank_match.group(1)) if six_month_rank_match else None

            # Extract Weighted Z-Score Rank
            weighted_rank_match = re.search(r'Weighted Z-Score Rank: (\d+) out of', content)
            weighted_rank = int(weighted_rank_match.group(1)) if weighted_rank_match else None

            # Extract RSI value
            rsi_match = re.search(r'RSI \(14-day\): ([\d.]+)', content)
            rsi = float(rsi_match.group(1)) if rsi_match else None

            return {
                'ticker': ticker,
                'company_name': company_name,
                'current_price': current_price,
                'one_year_change': one_year_change,
                'six_month_change': six_month_change,
                'one_year_rank': one_year_rank,
                'six_month_rank': six_month_rank,
                'weighted_rank': weighted_rank,
                'rsi': rsi
            }
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Function to check if a stock has positive momentum
def has_positive_momentum(stock):
    # Criteria for positive momentum:
    # - Positive 1-year and/or 6-month change
    # - RSI above 50 (if available)
    # - Good rank relative to peers

    criteria_met = 0
    criteria_total = 0

    if stock['one_year_change'] is not None:
        criteria_total += 1
        if stock['one_year_change'] > 0:
            criteria_met += 1

    if stock['six_month_change'] is not None:
        criteria_total += 1
        if stock['six_month_change'] > 0:
            criteria_met += 1

    if stock['rsi'] is not None:
        criteria_total += 1
        if stock['rsi'] > 50:  # RSI above 50 indicates bullish momentum
            criteria_met += 1

    if stock['one_year_rank'] is not None and stock['weighted_rank'] is not None:
        criteria_total += 1
        # Consider top 40% of stocks as having good ranking
        total_stocks = 250  # Approximate number of smallcap stocks
        if stock['weighted_rank'] <= (total_stocks * 0.4):
            criteria_met += 1

    # Require meeting at least 50% of the available criteria
    return criteria_total > 0 and (criteria_met / criteria_total) >= 0.5

# Main function to analyze smallcap stocks with positive momentum
def find_smallcap_with_positive_momentum():
    stock_data = []

    # Process all .md files in the smallcap directory
    for file_path in smallcap_dir.glob('*.md'):
        data = extract_stock_data(file_path)
        if data:
            stock_data.append(data)

    # Filter stocks with positive momentum
    positive_momentum_stocks = [stock for stock in stock_data if has_positive_momentum(stock)]

    # Sort by weighted rank (if available) or by 6-month change
    sorted_stocks = sorted(
        positive_momentum_stocks,
        key=lambda x: (
            x['weighted_rank'] if x['weighted_rank'] is not None else float('inf'),
            -1 * (x['six_month_change'] if x['six_month_change'] is not None else 0)
        )
    )

    # Print the header
    print(f"\n===== SMALLCAP STOCKS WITH POSITIVE MOMENTUM ({len(sorted_stocks)} stocks) =====")
    print(f"{'Rank':<5} {'Ticker':<10} {'Company':<30} {'Current Price':<15} {'1-Year %':<10} {'6-Month %':<10} {'RSI':<8} {'W-Rank':<8}")
    print("-" * 100)

    # Print the stocks with positive momentum
    for i, stock in enumerate(sorted_stocks):
        print(f"{i+1:<5} {stock['ticker']:<10} {stock['company_name'][:30]:<30} " +
              f"₹{stock['current_price']:<14,.2f} " +
              f"{stock['one_year_change']:<10.2f} " +
              f"{stock['six_month_change']:<10.2f} " +
              f"{stock['rsi']:<8.2f} " +
              f"{stock['weighted_rank']}")

if __name__ == "__main__":
    find_smallcap_with_positive_momentum()