import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from urllib.parse import parse_qs
import json
import pandas as pd
from datetime import datetime
import uuid
import os
from typing import Callable, Any
from wsgiref.simple_server import make_server

# Download necessary NLTK data
nltk.download('vader_lexicon', quiet=True)

# Initialize Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

# Load reviews from CSV file
reviews = pd.read_csv('data/reviews.csv').to_dict(orient='records')

# Define valid locations
VALID_LOCATIONS = [
    "Albuquerque, New Mexico", "Carlsbad, California", "Chula Vista, California", 
    "Colorado Springs, Colorado", "Denver, Colorado", "El Cajon, California",
    "El Paso, Texas", "Escondido, California", "Fresno, California",
    "La Mesa, California", "Las Vegas, Nevada", "Los Angeles, California",
    "Oceanside, California", "Phoenix, Arizona", "Sacramento, California",
    "Salt Lake City, Utah", "San Diego, California", "Tucson, Arizona"
]

class ReviewAnalyzerServer:
    def __init__(self) -> None:
        pass

    def analyze_sentiment(self, review_body: str) -> dict:
        return sia.polarity_scores(review_body)

    def __call__(self, environ: dict[str, Any], start_response: Callable[..., Any]) -> bytes:
        method = environ.get("REQUEST_METHOD")

        if method == "GET":
            return self.handle_get(environ, start_response)

        if method == "POST":
            return self.handle_post(environ, start_response)

        start_response("405 Method Not Allowed", [("Content-Type", "text/plain")])
        return [b"Method Not Allowed"]

    def handle_get(self, environ: dict[str, Any], start_response: Callable[..., Any]) -> bytes:
        query_params = parse_qs(environ.get('QUERY_STRING', ''))
        location = query_params.get('location', [None])[0]
        start_date = query_params.get('start_date', [None])[0]
        end_date = query_params.get('end_date', [None])[0]

        filtered_reviews = [
            {
                **review,
                "sentiment": self.analyze_sentiment(review['ReviewBody'])
            }
            for review in reviews
            if self.filter_review(review, location, start_date, end_date)
        ]

        filtered_reviews.sort(key=lambda x: x['sentiment']['compound'], reverse=True)

        response_body = json.dumps(filtered_reviews, indent=2).encode("utf-8")
        start_response("200 OK", [
            ("Content-Type", "application/json"),
            ("Content-Length", str(len(response_body)))
        ])
        return [response_body]

    def filter_review(self, review: dict, location: str, start_date: str, end_date: str) -> bool:
        review_date = datetime.strptime(review['Timestamp'], "%Y-%m-%d %H:%M:%S")

        if location and review['Location'] != location:
            return False
        if start_date and review_date < datetime.strptime(start_date, "%Y-%m-%d"):
            return False
        if end_date and review_date > datetime.strptime(end_date, "%Y-%m-%d"):
            return False

        return True

    def handle_post(self, environ: dict[str, Any], start_response: Callable[..., Any]) -> bytes:
        content_length = int(environ.get('CONTENT_LENGTH', 0))
        body = environ['wsgi.input'].read(content_length).decode('utf-8')
        post_data = parse_qs(body)

        location = post_data.get('Location', [None])[0]
        review_body = post_data.get('ReviewBody', [None])[0]

        if location not in VALID_LOCATIONS:
            start_response("400 Bad Request", [
                ("Content-Type", "application/json")
            ])
            return [b'{"error": "Invalid location."}']

        if not location or not review_body:
            start_response("400 Bad Request", [
                ("Content-Type", "application/json")
            ])
            return [b'{"error": "Location and ReviewBody are required."}']

        new_review = {
            "ReviewId": str(uuid.uuid4()),
            "ReviewBody": review_body,
            "Location": location,
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        reviews.append(new_review)

        response_body = json.dumps(new_review, indent=2).encode("utf-8")
        start_response("201 Created", [
            ("Content-Type", "application/json"),
            ("Content-Length", str(len(response_body)))
        ])
        return [response_body]

if __name__ == "__main__":
    app = ReviewAnalyzerServer()
    port = int(os.environ.get('PORT', 8000))
    with make_server("", port, app) as httpd:
        print(f"Listening on port {port}...")
        httpd.serve_forever()
