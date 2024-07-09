# Book Recommender - Flask Backend

This repository contains the backend of the Book Recommender application. The backend is built using Flask and provides APIs for the frontend to interact with. It uses a unique book recommendation system based on the dominant colors in book covers as well as book title similarity matching.

The Frontend Repository can be found here https://github.com/Fabreeeze/Book_recommender_react_frontend


## Features

- **Book Recommendation**: Recommends books based on the colors in their covers and title similarity .
- **Data Analysis**: Utilizes pandas and numpy for data manipulation.
- **Machine Learning**: Implements scikit-learn for similarity matching.
- **Backup Frontend**: Includes the original HTML, CSS, and JavaScript frontend as a backup.

## Technologies Used

- Flask
- Python
- pandas
- numpy
- scikit-learn
- openpyxl

## Backup Frontend

The original frontend built with HTML, CSS, and JavaScript is still available in this repository. This serves as a backup in case the React-based frontend does not work. You can access it by navigating to the `templates` and `static` directories.

## Setup Instructions

1. **Clone the repository**:
    ```bash
    git clone https://github.com/Fabreeeze/book-recommender-using-bookTitle-text-matching-and-book-cover-image-colour-analysis
    cd book-recommender-using-bookTitle-text-matching-and-book-cover-image-colour-analysis
    ```

2. **Create and activate a virtual environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Run the application**:
    ```bash
    flask run
    ```

## API Endpoints

- `POST /search` - Searches for the nearest matching book based on inputs specified.


