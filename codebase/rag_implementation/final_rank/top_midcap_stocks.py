import os
import re
from pathlib import Path

# Define the path to the midcap stocks directory
midcap_dir = Path(r"e:\Practise_Code\clean\stock_analysis\reports_v2\midcap_150")

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

            return {
                'ticker': ticker,
                'company_name': company_name,
                'current_price': current_price,
                'one_year_change': one_year_change,
                'six_month_change': six_month_change,
                'one_year_rank': one_year_rank,
                'six_month_rank': six_month_rank,
                'weighted_rank': weighted_rank
            }
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Main function to analyze all midcap stocks
def analyze_midcap_stocks():
    stock_data = []

    # Process all .md files in the midcap directory
    for file_path in midcap_dir.glob('*.md'):
        data = extract_stock_data(file_path)
        if data:
            stock_data.append(data)

    # Sort stocks by 1-Year Momentum Rank (lower rank is better)
    one_year_top = sorted(stock_data, key=lambda x: x['one_year_rank'] if x['one_year_rank'] is not None else float('inf'))

    # Sort stocks by 6-Month Momentum Rank
    six_month_top = sorted(stock_data, key=lambda x: x['six_month_rank'] if x['six_month_rank'] is not None else float('inf'))

    # Sort stocks by Weighted Z-Score Rank
    weighted_top = sorted(stock_data, key=lambda x: x['weighted_rank'] if x['weighted_rank'] is not None else float('inf'))

    # Print the top 10 performing stocks by 1-Year Momentum
    print("\n===== TOP 10 MIDCAP STOCKS BY 1-YEAR MOMENTUM =====")
    print(f"{'Rank':<5} {'Ticker':<10} {'Company':<30} {'Current Price':<15} {'1-Year Change':<15}")
    print("-" * 80)
    for i, stock in enumerate(one_year_top[:10]):
        print(f"{i+1:<5} {stock['ticker']:<10} {stock['company_name'][:30]:<30} ₹{stock['current_price']:<14,.2f} {stock['one_year_change']:<14,.2f}%")

    # Print the top 10 performing stocks by 6-Month Momentum
    print("\n===== TOP 10 MIDCAP STOCKS BY 6-MONTH MOMENTUM =====")
    print(f"{'Rank':<5} {'Ticker':<10} {'Company':<30} {'Current Price':<15} {'6-Month Change':<15}")
    print("-" * 80)
    for i, stock in enumerate(six_month_top[:10]):
        print(f"{i+1:<5} {stock['ticker']:<10} {stock['company_name'][:30]:<30} ₹{stock['current_price']:<14,.2f} {stock['six_month_change']:<14,.2f}%")

    # Print the top 10 performing stocks by Weighted Z-Score (combined performance)
    print("\n===== TOP 10 MIDCAP STOCKS BY WEIGHTED Z-SCORE (COMBINED PERFORMANCE) =====")
    print(f"{'Rank':<5} {'Ticker':<10} {'Company':<30} {'Current Price':<15} {'1-Year Change':<15} {'6-Month Change':<15}")
    print("-" * 95)
    for i, stock in enumerate(weighted_top[:10]):
        print(f"{i+1:<5} {stock['ticker']:<10} {stock['company_name'][:30]:<30} ₹{stock['current_price']:<14,.2f} {stock['one_year_change']:<14,.2f}% {stock['six_month_change']:<14,.2f}%")

if __name__ == "__main__":
    analyze_midcap_stocks()