# AI Hedge Fund

FORKED FROM [virattt ai-hedge-fund](https://github.com/virattt/ai-hedge-fund)

This project uses LangChain and LLMs to make automated trading decisions.

## Agents

*   **Market Data Agent:** Gathers and preprocesses market data, including historical prices, financial metrics, insider trades, general market data, and cash flow statements.
*   **Quantitative Agent:** Analyzes technical indicators such as MACD, RSI, Bollinger Bands, and OBV to generate trading signals.
*   **Fundamental Agent:** Analyzes fundamental data such as profitability, growth, financial health, and valuation ratios to generate trading signals.
*   **Sentiment Agent:** Analyzes market sentiment based on insider trades to generate trading signals.
*   **Risk Management Agent:** Evaluates portfolio risk and sets position limits based on the analysis from the other agents.
*   **Portfolio Management Agent:** Makes final trading decisions and generates orders based on the analysis from all other agents.

## How to run

1.  Navigate to the `ai-hedge-fund` directory:
    ```bash
    cd ai-hedge-fund
    ```
2.  Create a virtual environment:
    ```bash
    uv venv
    ```
3.  Install the project dependencies:
    ```bash
    uv pip install -e .
    ```
4.  Run the hedge fund:
    ```bash
    uv run src/agents.py --ticker AAPL --show-reasoning
    ```

    The `--show-reasoning` flag will print the reasoning from each agent, providing insights into the decision-making process.
