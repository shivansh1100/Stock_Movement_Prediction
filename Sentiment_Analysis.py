from transformers import pipeline


def classification(comment):
    classifier = pipeline('zero-shot-classification', framework='pt')
    p = classifier(comment, labels=["stocks", "positive", "negative"])
    return p


def sentiment(comment):
    classifier = pipeline("sentiment-analysis", model="yiyanghkust/finbert-tone", framework='pt')
    r = classifier(comment)
    return r

if __name__ == "__main__":
    print(sentiment("preeet is a good trader and has good investments"))