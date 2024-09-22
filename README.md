Please remember this program is still at its infancy the training has been done wth a very small data and the repo will be active till its proper implementation.
The repo being a project is focused on one single company stocks("NVIDIA") for simplicity purposes.


# Stock_Movement_Prediction
The following project showcases the stock movement on the bases of reddit comments and through stock news on google news. It comprises of 4 segments or files
Sentiment_Analysis.py
Scrape_it.py
Main.py
Final.ipynb
## Sentiment Analysis
Sentiment analysis is a simple pipelining of FinBERT model through Hugging Face(https://huggingface.co/). It provides with pretrained models adn FinBERT is one of Them

FineBERT is particuarly designed on financial related topics so it's a better model for sentiment analysis when it comes to stocks related topics.

Why FinBERT?
Bloomberg GPT can also be used only decision not to was the large parameters and large size = large computation and avoiding it still gives a lot better solutions.
### Classification
First function is Classification(). Zero Shot classification is pipelined to differentiate between nature of script and helps characterize and tag them better.
### Sentiment
Second function is sentiment where FinBERt is pipelined.

## Scrape_it
Best thing everrr!!

Scraping is so fun and more fun is to manupilate it to your needs.
### Search_Subreddits
First function contains search subreddits and it is done through Reddit API you can make your own through reddit's own website.(A scrape request limit may apply please read T&C before use).

It basically acts as a suggestion box, If i input 'NVIDIA' it will give list of all stocks that show up if you search it in reddit. For eg: NVDA_Stock etc.

### fetch_google_news

This took a lot of time to built adn it basically uses beautiful soup and based on html parsing finds the elements containing news aticles and date and time and scrape it back into a file.

This function was not required but due to individual bias, sarcasm, false news leads to unreliability in reddit or twitter chats that's why a reliable source was required although at a small level.

### red_dat_Scraper

This is the most important fuction which takes the list from the first function(Seatch_Subreddit) and loops them to scrape messages from posts and their comments.

*Note: The model cuts comments to 500 char length to avoid including ramblings and it increases computations and the number of posts to be scraped can also be set inside function.*

It will finaly create a csv file and csv file will also contain sentiment as the sentiment function is called on eac comment and is noted with tag 
[0] neutral
[1] positive
[-1] negative

## Main.py

Main.py is just a simple function called for both Scrape_it.py and Sentiment_Analysis.py(optional for testing purposes).

## Final.ipynb

The Final model works out the prediction section

### 1
The first step is to import both reddit commetns and google_news. The google_news csv is not characterised by sentiment so the csv is formatted in first section of code where the selective rows are chosen and few changes are made for simplicity.

### 2
For sentiment analysis of google_news, bert_uncased is used, there is not specific reason is to why, it was used because googe news simply states the fact so there's no need for advance computations for bias check and other biases. 

But user can chaneg to any model they please.

Then the sentiments are converted to tags
### 3

The tags are 
[0] - Neutral
[1] - Positive
[2] - Negative

This helps in characterising the posts and helps in letting model know in what favor will the stocks go.

( It's better to assume some states but there should be someting kept in mind if one is seling stock, 100's are holding or investing.)

### 4

After the sentiments are found both files are joined together and formed as a cluster of  tags containing revies as to where the stocks might lead to.

### 5 
The train andn test data is divided and then the model is tested on random forest classifier and build the cluster classifiction in order to map the trends and predict one or two dat mapping based on that same idea.






