from concurrent.futures import ThreadPoolExecutor, as_completed
import Scrape_it as scr



#       \Scraping from Google_News
if __name__ == "__main__":
    # fetch_google_news('india')
    nifty_50_companies = [
        "Adani Enterprises (ADANIENT)",
        "Adani Ports and Special Economic Zone (ADANIPORTS)",
        "Asian Paints (ASIANPAINT)",
        "Axis Bank (AXISBANK)",
        "Bajaj Auto (BAJAJ-AUTO)",
        "Bajaj Finance (BAJFINANCE)",
        "Bajaj Finserv (BAJAJFINSV)",
        "Bharti Airtel (BHARTIARTL)",
        "BPCL (BPCL)",
        "Britannia Industries (BRITANNIA)",
        "Cipla (CIPLA)",
        "Coal India (COALINDIA)",
        "Divi's Laboratories (DIVISLAB)",
        "Dr. Reddy's Laboratories (DRREDDY)",
        "Eicher Motors (EICHERMOT)",
        "Grasim Industries (GRASIM)",
        "HCL Technologies (HCLTECH)",
        "HDFC (HDFC)",
        "HDFC Bank (HDFCBANK)",
        "HDFC Life Insurance (HDFCLIFE)",
        "Hero MotoCorp (HEROMOTOCO)",
        "Hindalco Industries (HINDALCO)",
        "Hindustan Unilever (HINDUNILVR)",
        "ICICI Bank (ICICIBANK)",
        "IndusInd Bank (INDUSINDBK)",
        "Infosys (INFY)",
        "ITC (ITC)",
        "JSW Steel (JSWSTEEL)",
        "Kotak Mahindra Bank (KOTAKBANK)",
        "Larsen & Toubro (LT)",
        "Mahindra & Mahindra (M&M)",
        "Maruti Suzuki (MARUTI)",
        "Nestle India (NESTLEIND)",
        "NTPC (NTPC)",
        "Oil and Natural Gas Corporation (ONGC)",
        "Power Grid Corporation of India (POWERGRID)",
        "Reliance Industries (RELIANCE)",
        "SBI Life Insurance (SBILIFE)",
        "State Bank of India (SBIN)",
        "Sun Pharmaceuticals (SUNPHARMA)",
        "Tata Consultancy Services (TCS)",
        "Tata Consumer Products (TATACONSUM)",
        "Tata Motors (TATAMOTORS)",
        "Tata Steel (TATASTEEL)",
        "Tech Mahindra (TECHM)",
        "Titan Company (TITAN)",
        "Ultratech Cement (ULTRACEMCO)",
        "UPL (UPL)",
        "Wipro (WIPRO)",
        "Nvidia (NVDA)"
    ]
    #
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(scr.fetch_google_news, company) for company in nifty_50_companies]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"An error occurred: {e}")

#Scraping from Reddit
if __name__ == '__main__':
    subs = scr.search_subreddits('NVDA')[:10]
    scr.red_dat_scraper(subs)
