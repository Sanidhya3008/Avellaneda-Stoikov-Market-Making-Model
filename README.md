# Avellaneda-Stoikov Market Making Model Simulator

This application implements the Avellaneda-Stoikov model for high-frequency market making in limit order books. It provides interactive visualizations and simulations to explore how different parameters affect market making performance.

https://market-makers.streamlit.app/

## Features

- **Interactive Simulations**: Run simulations of the Avellaneda-Stoikov model with customizable parameters
- **Strategy Comparison**: Compare the performance of different market making strategies
- **Parameter Sensitivity Analysis**: Explore how different parameter combinations affect profitability and risk
- **Visualization Tools**: View interactive plots of price evolution, inventory, and more
- **Educational Resources**: Learn about market making theory and the Avellaneda-Stoikov model

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone this repository:

```bash
git clone <repository-url>
cd avellaneda-stoikov-simulator
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the application:

```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`.

## Usage

1. Navigate to the "Model Parameters" section to set up the simulation parameters
2. Explore the "Visualizations" section to understand how the model works
3. Run a single simulation in the "Run Simulation" section
4. Compare different strategies in the "Strategy Comparison" section

## Model Background

The Avellaneda-Stoikov model provides a mathematical framework for market making in electronic limit order books. The key components include:

1. **Mid-Price Evolution**: The stock's mid-price follows a Brownian motion
2. **Reservation Price**: The market maker's personal valuation adjusted for inventory
3. **Order Arrivals**: Buy and sell orders arrive as Poisson processes with intensity depending on quote attractiveness
4. **Utility Maximization**: The market maker aims to maximize expected utility of terminal wealth

## Deployment

The application can be deployed in several ways:

- **Streamlit Cloud**: Push to GitHub and deploy via Streamlit Cloud
- **Heroku**: Use the included Procfile and setup.sh
- **Docker**: Build and run the Docker image using the included Dockerfile

## Course Alignment

This application aligns with the CSE3171 "Introduction to Simulation & Modeling" course, covering:

- Stochastic differential equations (SDEs) and simulation techniques
- Monte Carlo methods
- Random number generation
- Analysis of simulation data
- Queueing models and Poisson processes

## References

Avellaneda, M., & Stoikov, S. (2008). High-frequency trading in a limit order book. Quantitative Finance, 8(3), 217-224.
