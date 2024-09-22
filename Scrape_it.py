import csv
import sys

import praw
import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import Sentiment_Analysis as Sent


# Reddit API credentials
client_id = 'your_client_id'
client_secret = 'your_client_secret'
username = 'your_username'
password = 'your_password'

# Set up Reddit API connection
reddit = praw.Reddit(
    client_id='id',
    client_secret='sec',
    user_agent='api',
    username='id',
    password='pass')

def search_subreddits(query):
    # Searching subreddits based on the input query
    subreddits = reddit.subreddits.search(query)

    # Store results in a list
    result_list = []
    for subreddit in subreddits:
        result_list.append(subreddit.display_name)

    # Return the list of subreddits
    return result_list
def fetch_google_news(topic):
    url = f"https://news.google.com/search?q={topic.replace(' ', '%20')}&hl=en-US&gl=US&ceid=US%3Aen"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print(f"Failed to fetch news for topic: {topic}")
        return

    soup = BeautifulSoup(response.content, 'html.parser')
    articles = soup.find_all('article', class_='IFHyqb DeXSAc')


    news_data = []
    for article in articles:
        headline = article.find('a', class_='JtKRv')
        if headline:
            headline_text = headline.get_text()


            time_element = article.find('time', class_='hvbAAd')
            if time_element and time_element.has_attr('datetime'):
                timestamp = time_element['datetime']
                time_published = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                formatted_time = time_published.strftime("%Y-%m-%d %H:%M:%S")
            else:
                formatted_time = "Unknown"

            news_data.append({
                'topic': topic,
                'headline': headline_text,
                'time': formatted_time
            })

    df = pd.DataFrame(news_data)
    df = df.drop_duplicates(subset=['headline'])

    print(df)

    csv_file = '../sep_pro/nifty_50_companies_news.csv'
    if os.path.isfile(csv_file):
        df.to_csv(csv_file, mode='a', header=False, index=False)
    else:
        df.to_csv(csv_file, mode='w', header=True, index=False)

def red_dat_scraper(subss):
    subreddits = reddit.subreddits.search(subss)
    with open('Comments_Sentiment(20-09-24).csv', 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Date', 'Subreddit', 'Post Title', 'Post Text', 'Accuracy', 'Tag'])  # Header row
        for c_name in subss:
            sub = reddit.subreddit(c_name)
            hot_posts = sub.hot(limit=50)  # Scrape hot posts
            for post in hot_posts:
                # Extract post data
                pd = post.created_utc
                date_time = datetime.utcfromtimestamp(pd)
                p_date = date_time.strftime('%Y-%m-%d %H:%M:%S')
                p_title = post.title
                p_text = post.selftext
                comments = []  # Initialize comments list
                sc = []
                Tag = []
                for comment in post.comments.list():  # Scrape comments
                    if isinstance(comment, praw.models.MoreComments):  # Skip MoreComments objects
                        continue
                    if 'Nvidia' in comment.body:
                        comments.append(comment.body)
                a = ', '.join(comments)[:500]
                sc.append(Sent.sentiment(a))
                for ele in sc:
                    if ele[0]['label'] == 'Neutral':
                        Tag.append(0)
                    if ele[0]['label'] == 'Positive':
                        Tag.append(1)
                    if ele[0]['label'] == 'Negative':
                        Tag.append(-1)


                # for ele in sc:
                #     if ele.label == "Neutral":
                #         ele.pop()

                # Write data to CSV file
                writer.writerow([p_date, c_name, p_title, p_text[:30], sc, Tag])








