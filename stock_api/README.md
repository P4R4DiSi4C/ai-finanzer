# Stock API

This project is a simple API for retrieving stock data.

## How to run

1.  Navigate to the `stock_api` directory:
    ```bash
    cd stock_api
    ```
2.  Create a virtual environment:
    ```bash
    uv venv
    ```
3.  Install the project dependencies:
    ```bash
    uv pip install -e .
    ```
4.  Populate the database:
    ```bash
    uv run app/scripts/populate_db.py
    ```
5.  Run the API:
    ```bash
    uv run app/main.py
