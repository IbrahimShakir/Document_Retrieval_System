import requests
from bs4 import BeautifulSoup
from models import db, Document

def scrape_news_articles():
    """
    Scrape news articles from a sample website and store them in the database.
    """
    # Example of scraping logic (Replace with actual site and logic)
    response = requests.get("https://example.com/news")
    soup = BeautifulSoup(response.content, 'html.parser')

    # Assuming articles are in <article> tags
    articles = soup.find_all('article')

    for article in articles:
        content = article.get_text()
        # Example vectorization (Replace with actual encoding logic)
        vector = np.random.rand(512)  # Dummy vector
        doc = Document(content=content, vector=vector)
        db.session.add(doc)
    db.session.commit()

if __name__ == "__main__":
    with app.app_context():
        scrape_news_articles()
