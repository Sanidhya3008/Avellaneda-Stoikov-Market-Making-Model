import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime
import io
import base64

# Set page config
st.set_page_config(
    page_title="Avellaneda-Stoikov Market Making Model",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styling constants
MAIN_COLOR = '#1a5276'
SECONDARY_COLOR = '#2980b9'
ACCENT_COLOR = '#3498db'
LIGHT_COLOR = '#ebf5fb'
TEXT_COLOR = '#333333'
BACKGROUND_COLOR = '#f9f9f9'
GRID_COLOR = '#e0e0e0'

# Cache for storing simulation results
class SimulationCache:
    def __init__(self):
        self.last_params = {}
        self.results = {}
        self.figs = {}

simulation_cache = SimulationCache()

# Implementation of the Avellaneda-Stoikov market making model
class AvellanedaStoikovModel:
    """
    Implementation of the Avellaneda-Stoikov market making model.

    The model simulates the optimal bid and ask quotes for a market maker in a limit order book.
    It uses a stochastic differential equation (SDE) to model the mid-price evolution
    and incorporates risk aversion and inventory management.
    """

    def __init__(self, S0=100.0, T=1.0, sigma=2.0, M=1000, gamma=0.1,
                 k=1.5, A=140, I=1, seed=None):
        """
        Initialize the model with given parameters.

        Parameters:
        -----------
        S0 : float
            Initial mid-price
        T : float
            Time horizon in days
        sigma : float
            Volatility of the mid-price
        M : int
            Number of time steps
        gamma : float
            Risk aversion parameter
        k : float
            Shape parameter for order arrival intensity
        A : float
            Base intensity of order arrivals
        I : int
            Number of instruments
        seed : int or None
            Random seed for reproducibility
        """
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)

        # Model parameters
        self.S0 = S0
        self.T = T
        self.sigma = sigma
        self.M = M
        self.dt = T / M
        self.sqrt_dt = np.sqrt(self.dt)
        self.gamma = gamma
        self.k = k
        self.A = A
        self.I = I

        # Initialize arrays for simulation results
        self.time_grid = np.linspace(0, T, M+1)
        self.S = np.zeros((M+1, I))  # Mid-price
        self.Bid = np.zeros((M+1, I))  # Bid price
        self.Ask = np.zeros((M+1, I))  # Ask price
        self.ReservPrice = np.zeros((M+1, I))  # Reservation price
        self.spread = np.zeros((M+1, I))  # Bid-ask spread
        self.deltaB = np.zeros((M+1, I))  # Distance from mid to bid
        self.deltaA = np.zeros((M+1, I))  # Distance from ask to mid
        self.q = np.zeros((M+1, I))  # Inventory position
        self.w = np.zeros((M+1, I))  # Wealth
        self.equity = np.zeros((M+1, I))  # Total equity (wealth + position value)
        self.trade_times_bid = []  # Times when bid orders are executed
        self.trade_times_ask = []  # Times when ask orders are executed

        # Initial values
        self.S[0] = S0
        self.ReservPrice[0] = S0
        self.Bid[0] = S0
        self.Ask[0] = S0

    def simulate_mid_price(self):
        """
        Simulate the mid-price evolution using Geometric Brownian Motion (GBM).
        This is an implementation of a Stochastic Differential Equation (SDE).

        The SDE is of the form: dS = σ·S·dW, where W is a Wiener process.
        We use the Euler-Maruyama method to simulate this SDE.
        """
        for t in range(1, self.M+1):
            # Generate random normal increments for the Wiener process
            z = np.random.standard_normal(self.I)

            # Euler-Maruyama method for the SDE
            self.S[t] = self.S[t-1] + self.sigma * self.sqrt_dt * z

    def calculate_quotes(self):
        """
        Calculate optimal bid and ask quotes based on the model.
        """
        for t in range(1, self.M+1):
            # Calculate reservation price (adjusted for inventory risk)
            # r(s,t) = s - q * gamma * sigma**2 * (T-t)
            remaining_time = self.T - t * self.dt
            self.ReservPrice[t] = self.S[t] - self.q[t-1] * self.gamma * (self.sigma ** 2) * remaining_time

            # Calculate optimal spread
            self.spread[t] = self.gamma * (self.sigma ** 2) * remaining_time + (2/self.gamma) * np.log(1 + (self.gamma/self.k))

            # Set bid and ask prices
            self.Bid[t] = self.ReservPrice[t] - self.spread[t]/2
            self.Ask[t] = self.ReservPrice[t] + self.spread[t]/2

            # Calculate distance from mid-price
            self.deltaB[t] = self.S[t] - self.Bid[t]
            self.deltaA[t] = self.Ask[t] - self.S[t]

    def simulate_order_arrivals(self):
        """
        Simulate order arrivals and executions based on intensities.
        Uses a Poisson process with intensity depending on quote distance from mid-price.
        """
        for t in range(1, self.M+1):
            # Calculate order arrival intensities
            lambdaA = self.A * np.exp(-self.k * self.deltaA[t])
            lambdaB = self.A * np.exp(-self.k * self.deltaB[t])

            # Calculate probability of order arrivals in this time step
            ProbA = 1 - np.exp(-lambdaA * self.dt)
            ProbB = 1 - np.exp(-lambdaB * self.dt)

            # Generate random numbers for order execution
            fa = np.random.random()
            fb = np.random.random()

            # Default: no position or wealth change
            self.q[t] = self.q[t-1]
            self.w[t] = self.w[t-1]

            # Check for order executions
            if ProbB > fb and ProbA < fa:
                # Buy market order hits our bid
                self.q[t] += 1
                self.w[t] -= self.Bid[t]
                self.trade_times_bid.append(t)

            if ProbB < fb and ProbA > fa:
                # Sell market order hits our ask
                self.q[t] -= 1
                self.w[t] += self.Ask[t]
                self.trade_times_ask.append(t)

            if ProbB > fb and ProbA > fa:
                # Both sides executed (rare but possible)
                self.w[t] = self.w[t] - self.Bid[t] + self.Ask[t]
                self.trade_times_bid.append(t)
                self.trade_times_ask.append(t)

            # Calculate equity at this time step
            self.equity[t] = self.w[t] + self.q[t] * self.S[t]

    def run_simulation(self):
        """
        Run a complete simulation of the model.
        """
        self.simulate_mid_price()
        self.calculate_quotes()
        self.simulate_order_arrivals()

    def get_results(self):
        """
        Return key results from the simulation.
        """
        return {
            'final_equity': self.equity[-1][0],
            'avg_spread': np.mean(self.spread),
            'max_inventory': np.max(np.abs(self.q)),
            'final_inventory': self.q[-1][0],
            'std_equity': np.std(self.equity)
        }

    def generate_plotly_graph(self):
        """
        Generate interactive Plotly visualizations for simulation results.
        """
        # Create a subplot with 2 rows and 2 columns
        fig = make_subplots(
            rows=2, cols=2, 
            subplot_titles=(
                'Price Evolution', 
                'Inventory Evolution', 
                'PnL Evolution', 
                'Spread Evolution'
            ),
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )
        
        # Plot 1: Prices
        fig.add_trace(
            go.Scatter(
                x=self.time_grid, 
                y=self.S[:, 0], 
                mode='lines', 
                name='Mid Price',
                line=dict(color='rgb(31, 119, 180)', width=2)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=self.time_grid, 
                y=self.Ask[:, 0], 
                mode='lines', 
                name='Ask',
                line=dict(color='rgb(255, 127, 14)', width=1.5, dash='dot')
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=self.time_grid, 
                y=self.Bid[:, 0], 
                mode='lines', 
                name='Bid',
                line=dict(color='rgb(44, 160, 44)', width=1.5, dash='dot')
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=self.time_grid, 
                y=self.ReservPrice[:, 0], 
                mode='lines', 
                name='Reservation Price',
                line=dict(color='rgb(214, 39, 40)', width=1.5, dash='dashdot')
            ),
            row=1, col=1
        )

        # Add markers for trades
        bid_times = [self.time_grid[t] for t in self.trade_times_bid]
        bid_prices = [self.Bid[t, 0] for t in self.trade_times_bid]
        
        ask_times = [self.time_grid[t] for t in self.trade_times_ask]
        ask_prices = [self.Ask[t, 0] for t in self.trade_times_ask]
        
        fig.add_trace(
            go.Scatter(
                x=bid_times, 
                y=bid_prices, 
                mode='markers', 
                name='Bid Executed',
                marker=dict(color='green', size=8, symbol='triangle-up')
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=ask_times, 
                y=ask_prices, 
                mode='markers', 
                name='Ask Executed',
                marker=dict(color='red', size=8, symbol='triangle-down')
            ),
            row=1, col=1
        )
        
        # Plot 2: Inventory
        fig.add_trace(
            go.Scatter(
                x=self.time_grid, 
                y=self.q[:, 0], 
                mode='lines', 
                name='Inventory',
                line=dict(color='rgb(148, 103, 189)', width=2)
            ),
            row=1, col=2
        )
        
        # Plot 3: PnL
        fig.add_trace(
            go.Scatter(
                x=self.time_grid, 
                y=self.equity[:, 0], 
                mode='lines', 
                name='PnL',
                line=dict(color='rgb(140, 86, 75)', width=2)
            ),
            row=2, col=1
        )
        
        # Plot 4: Spread
        fig.add_trace(
            go.Scatter(
                x=self.time_grid, 
                y=self.spread[:, 0], 
                mode='lines', 
                name='Spread',
                line=dict(color='rgb(227, 119, 194)', width=2)
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            width=1000,
            title_text='Avellaneda-Stoikov Model Simulation Results',
            template='plotly_white',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Update axes labels
        fig.update_xaxes(title_text='Time', row=2, col=1)
        fig.update_xaxes(title_text='Time', row=2, col=2)
        fig.update_yaxes(title_text='Price', row=1, col=1)
        fig.update_yaxes(title_text='Position', row=1, col=2)
        fig.update_yaxes(title_text='Equity', row=2, col=1)
        fig.update_yaxes(title_text='Spread', row=2, col=2)
        
        return fig

def run_multiple_simulations(num_sims=100, **params):
    """
    Run multiple simulations and collect statistics.

    Parameters:
    -----------
    num_sims : int
        Number of simulations to run
    **params : dict
        Parameters to pass to the AvellanedaStoikovModel constructor

    Returns:
    --------
    pd.DataFrame
        DataFrame with statistics from all simulations
    """
    results = []

    # Run simulations
    for i in range(num_sims):
        model = AvellanedaStoikovModel(**params)
        model.run_simulation()
        result = model.get_results()
        results.append(result)

    return pd.DataFrame(results)

def compare_strategies(S0=100.0, T=1.0, sigma=2.0, M=1000, gamma=0.1, k=1.5, A=140, num_sims=100):
    """
    Compare different market making strategies and return the results.
    """
    # Dictionary to store results
    strategy_results = {}
    
    # Calculate the optimal spread for inventory strategy
    optimal_spread = (2/gamma) * np.log(1 + (gamma/k))
    
    # Dictionary to store simulation results
    results_inventory = []
    results_symmetric = []
    results_best_bid_ask = []
    
    # Market spread (typically smaller than optimal spread)
    market_spread = 0.5
    
    # Run simulations
    for i in range(num_sims):
        # Set seed for each simulation to ensure fair comparison
        seed = 42 * i
        
        # 1. Inventory Strategy
        model_inventory = AvellanedaStoikovModel(
            S0=S0, T=T, sigma=sigma, M=M, gamma=gamma, k=k, A=A, seed=seed
        )
        model_inventory.run_simulation()
        result_inventory = model_inventory.get_results()
        result_inventory['strategy'] = 'Inventory'
        results_inventory.append(result_inventory)
        
        # 2. Symmetric Strategy (same spread as inventory but centered on mid-price)
        # We need to implement this strategy by modifying the calculate_quotes method
        model_symmetric = AvellanedaStoikovModel(
            S0=S0, T=T, sigma=sigma, M=M, gamma=gamma, k=k, A=A, seed=seed
        )
        model_symmetric.simulate_mid_price()  # Same price path
        
        # Override quote calculation for symmetric strategy
        for t in range(1, model_symmetric.M+1):
            # Use same spread as inventory strategy but center it on mid-price
            model_symmetric.spread[t] = optimal_spread
            model_symmetric.Bid[t] = model_symmetric.S[t] - optimal_spread/2
            model_symmetric.Ask[t] = model_symmetric.S[t] + optimal_spread/2
            model_symmetric.deltaB[t] = model_symmetric.S[t] - model_symmetric.Bid[t]
            model_symmetric.deltaA[t] = model_symmetric.Ask[t] - model_symmetric.S[t]
            model_symmetric.ReservPrice[t] = model_symmetric.S[t]  # Not used but for consistency
        
        model_symmetric.simulate_order_arrivals()  # Use same order arrival simulation
        result_symmetric = model_symmetric.get_results()
        result_symmetric['strategy'] = 'Symmetric'
        results_symmetric.append(result_symmetric)
        
        # 3. Best Bid/Best Ask Strategy (uses market spread)
        model_best_bid_ask = AvellanedaStoikovModel(
            S0=S0, T=T, sigma=sigma, M=M, gamma=gamma, k=k, A=A, seed=seed
        )
        model_best_bid_ask.simulate_mid_price()  # Same price path
        
        # Override quote calculation for best bid/ask strategy
        for t in range(1, model_best_bid_ask.M+1):
            model_best_bid_ask.spread[t] = market_spread
            model_best_bid_ask.Bid[t] = model_best_bid_ask.S[t] - market_spread/2
            model_best_bid_ask.Ask[t] = model_best_bid_ask.S[t] + market_spread/2
            model_best_bid_ask.deltaB[t] = model_best_bid_ask.S[t] - model_best_bid_ask.Bid[t]
            model_best_bid_ask.deltaA[t] = model_best_bid_ask.Ask[t] - model_best_bid_ask.S[t]
            model_best_bid_ask.ReservPrice[t] = model_best_bid_ask.S[t]  # Not used but for consistency
        
        model_best_bid_ask.simulate_order_arrivals()  # Use same order arrival simulation
        result_best_bid_ask = model_best_bid_ask.get_results()
        result_best_bid_ask['strategy'] = 'Best Bid/Ask'
        results_best_bid_ask.append(result_best_bid_ask)
    
    # Combine all results
    all_results = results_inventory + results_symmetric + results_best_bid_ask
    results_df = pd.DataFrame(all_results)
    
    # Calculate summary statistics for each strategy
    strategy_results = results_df.groupby('strategy').agg({
        'final_equity': ['mean', 'std'],
        'avg_spread': 'mean',
        'max_inventory': ['mean', 'max'],
        'final_inventory': ['mean', 'std'],
        'std_equity': 'mean'
    })
    
    # Flatten column names
    strategy_results.columns = ['_'.join(col).strip('_') for col in strategy_results.columns.values]
    
    # Create histograms of profits for each strategy
    fig_histograms = create_profit_histograms(results_df)
    
    return strategy_results, fig_histograms, results_df

def create_profit_histograms(results_df):
    """
    Create histograms of profits for each strategy
    """
    # Create a figure with subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            'Profit Distribution by Strategy', 
            'Risk-Return Profile'
        ),
        specs=[[{"type": "histogram"}, {"type": "scatter"}]]
    )
    
    # Colors for each strategy
    colors = {
        'Inventory': 'rgb(31, 119, 180)',
        'Symmetric': 'rgb(255, 127, 14)',
        'Best Bid/Ask': 'rgb(44, 160, 44)'
    }
    
    # Create histogram for each strategy
    for strategy in results_df['strategy'].unique():
        strategy_data = results_df[results_df['strategy'] == strategy]
        
        fig.add_trace(
            go.Histogram(
                x=strategy_data['final_equity'],
                name=strategy,
                marker_color=colors[strategy],
                opacity=0.7,
                nbinsx=30
            ),
            row=1, col=1
        )
    
    # Create scatter plot of risk vs return
    for strategy in results_df['strategy'].unique():
        strategy_data = results_df[results_df['strategy'] == strategy]
        
        fig.add_trace(
            go.Scatter(
                x=strategy_data['std_equity'],
                y=strategy_data['final_equity'],
                mode='markers',
                name=strategy,
                marker=dict(
                    color=colors[strategy],
                    size=8,
                    opacity=0.6
                )
            ),
            row=1, col=2
        )
        
        # Add a point for the mean of each strategy
        fig.add_trace(
            go.Scatter(
                x=[strategy_data['std_equity'].mean()],
                y=[strategy_data['final_equity'].mean()],
                mode='markers',
                name=f'{strategy} (Mean)',
                marker=dict(
                    color=colors[strategy],
                    size=15,
                    line=dict(
                        color='white',
                        width=2
                    ),
                    symbol='star'
                ),
                showlegend=False
            ),
            row=1, col=2
        )
    
    # Update layout
    fig.update_layout(
        height=500,
        width=1000,
        title_text='Strategy Comparison',
        template='plotly_white',
        barmode='overlay',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Update axes labels
    fig.update_xaxes(title_text='Final Equity', row=1, col=1)
    fig.update_yaxes(title_text='Frequency', row=1, col=1)
    fig.update_xaxes(title_text='Risk (Standard Deviation of Equity)', row=1, col=2)
    fig.update_yaxes(title_text='Return (Final Equity)', row=1, col=2)
    
    return fig

def generate_heatmap_analysis(gamma_values, k_values, S0=100.0, T=1.0, sigma=2.0, M=1000, A=140):
    """
    Generate a heatmap analysis of different gamma and k parameter combinations.
    
    Parameters:
    -----------
    gamma_values : list
        List of gamma values to test
    k_values : list
        List of k values to test
    
    Returns:
    --------
    go.Figure
        Plotly heatmap figure showing parameter sensitivity
    """
    # Grid for results
    results_grid = np.zeros((len(gamma_values), len(k_values)))
    risk_grid = np.zeros((len(gamma_values), len(k_values)))
    spread_grid = np.zeros((len(gamma_values), len(k_values)))
    
    # Run simulations for each parameter combination
    for i, gamma in enumerate(gamma_values):
        for j, k in enumerate(k_values):
            # Run a single simulation for each parameter set (multiple would be better but slower)
            model = AvellanedaStoikovModel(
                S0=S0, T=T, sigma=sigma, M=M, gamma=gamma, k=k, A=A, seed=42
            )
            model.run_simulation()
            results = model.get_results()
            
            # Store results
            results_grid[i, j] = results['final_equity']
            risk_grid[i, j] = results['std_equity']
            spread_grid[i, j] = results['avg_spread']
    
    # Create subplots for different metrics
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=(
            'Final Equity by Parameters',
            'Risk (Std of Equity) by Parameters',
            'Average Spread by Parameters'
        ),
        specs=[[{"type": "heatmap"}, {"type": "heatmap"}, {"type": "heatmap"}]]
    )
    
    # Create heatmap for final equity
    fig.add_trace(
        go.Heatmap(
            z=results_grid,
            x=[str(round(k, 2)) for k in k_values],
            y=[str(round(gamma, 2)) for gamma in gamma_values],
            colorscale='Viridis',
            colorbar=dict(title='Final Equity', x=0.3)
        ),
        row=1, col=1
    )
    
    # Create heatmap for risk
    fig.add_trace(
        go.Heatmap(
            z=risk_grid,
            x=[str(round(k, 2)) for k in k_values],
            y=[str(round(gamma, 2)) for gamma in gamma_values],
            colorscale='Reds',
            colorbar=dict(title='Risk', x=0.64)
        ),
        row=1, col=2
    )
    
    # Create heatmap for spread
    fig.add_trace(
        go.Heatmap(
            z=spread_grid,
            x=[str(round(k, 2)) for k in k_values],
            y=[str(round(gamma, 2)) for gamma in gamma_values],
            colorscale='Blues',
            colorbar=dict(title='Avg Spread', x=0.98)
        ),
        row=1, col=3
    )
    
    # Update layout
    fig.update_layout(
        height=400,
        width=1000,
        title_text='Parameter Sensitivity Analysis',
        template='plotly_white'
    )
    
    # Update axes labels
    for i in range(1, 4):
        fig.update_xaxes(title_text='Order Intensity Shape (k)', row=1, col=i)
        fig.update_yaxes(title_text='Risk Aversion (γ)', row=1, col=i)
    
    return fig

def plot_reservation_price_surface(gamma=0.1, sigma=2.0, T=1.0):
    """
    Create a 3D visualization of how reservation price changes with inventory and time.
    """
    # Create a grid of inventory values and time values
    q_values = np.linspace(-10, 10, 21)
    t_values = np.linspace(0, T, 20)
    Q, T_mesh = np.meshgrid(q_values, t_values)
    
    # Calculate reservation price for each point in the grid
    # r(s,q,t) = s - q * gamma * sigma^2 * (T-t)
    R = np.zeros_like(Q)
    S0 = 100.0
    for i in range(len(t_values)):
        for j in range(len(q_values)):
            remaining_time = T - t_values[i]
            R[i, j] = S0 - q_values[j] * gamma * (sigma ** 2) * remaining_time
    
    # Create interactive 3D surface plot
    fig = go.Figure(data=[
        go.Surface(
            z=R,
            x=T_mesh,
            y=Q,
            colorscale='Viridis',
            opacity=0.8
        )
    ])
    
    # Update layout
    fig.update_layout(
        title='Reservation Price as a Function of Time and Inventory',
        scene=dict(
            xaxis_title='Time',
            yaxis_title='Inventory',
            zaxis_title='Reservation Price',
            camera=dict(eye=dict(x=1.5, y=-1.5, z=1)),
            aspectratio=dict(x=1, y=1, z=0.7)
        ),
        width=800,
        height=600
    )
    
    return fig

def plot_order_arrival_intensity(k_values=[0.5, 1.0, 1.5, 2.0, 3.0], A=140, delta_max=5.0):
    """
    Visualize how order arrival intensity changes with distance to mid-price.
    """
    # Create distance grid
    delta = np.linspace(0, delta_max, 100)
    
    # Create figure
    fig = go.Figure()
    
    # Add traces for each k value
    for k in k_values:
        # Calculate intensity using the exponential formula from the paper
        intensity = A * np.exp(-k * delta)
        
        fig.add_trace(
            go.Scatter(
                x=delta,
                y=intensity,
                mode='lines',
                name=f'k = {k}',
                line=dict(width=2)
            )
        )
    
    # Update layout
    fig.update_layout(
        title='Order Arrival Intensity vs Distance from Mid-Price',
        xaxis_title='Distance from Mid-Price (δ)',
        yaxis_title='Arrival Intensity λ(δ)',
        template='plotly_white',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        ),
        width=700,
        height=400
    )
    
    # Add annotation explaining the formula
    fig.add_annotation(
        x=delta_max * 0.7,
        y=A * 0.8,
        text="λ(δ) = A * e^(-k*δ)",
        showarrow=False,
        font=dict(size=14)
    )
    
    return fig

def plot_optimal_quotes_by_inventory(gamma=0.1, sigma=2.0, T=1.0, k=1.5, q_range=(-10, 10)):
    """
    Visualize how optimal quotes change with inventory level.
    """
    # Create inventory grid
    q_values = np.arange(q_range[0], q_range[1] + 1, 1)
    
    # Calculate fixed spread component based on the paper
    fixed_spread = (2/gamma) * np.log(1 + (gamma/k))
    half_spread = fixed_spread / 2
    
    # Calculate reservation price, bid and ask adjustments for each inventory level
    # Assuming current time is t=0, so remaining time is T
    reservation_adjustments = -gamma * (sigma**2) * T * q_values
    bid_adjustments = reservation_adjustments - half_spread
    ask_adjustments = reservation_adjustments + half_spread
    
    # Create figure
    fig = go.Figure()
    
    # Add bid adjustment
    fig.add_trace(
        go.Scatter(
            x=q_values,
            y=bid_adjustments,
            mode='lines+markers',
            name='Bid Adjustment',
            line=dict(color='green', width=2)
        )
    )
    
    # Add reservation price adjustment
    fig.add_trace(
        go.Scatter(
            x=q_values,
            y=reservation_adjustments,
            mode='lines+markers',
            name='Reservation Price',
            line=dict(color='blue', width=2)
        )
    )
    
    # Add ask adjustment
    fig.add_trace(
        go.Scatter(
            x=q_values,
            y=ask_adjustments,
            mode='lines+markers',
            name='Ask Adjustment',
            line=dict(color='red', width=2)
        )
    )
    
    # Add zero line
    fig.add_shape(
        type="line",
        x0=q_range[0],
        y0=0,
        x1=q_range[1],
        y1=0,
        line=dict(
            color="gray",
            width=1,
            dash="dash",
        )
    )
    
    # Update layout
    fig.update_layout(
        title='Optimal Quote Adjustments by Inventory Level',
        xaxis_title='Inventory Position (q)',
        yaxis_title='Price Adjustment from Mid-Price',
        template='plotly_white',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        ),
        width=700,
        height=400
    )
    
    # Add annotations
    fig.add_annotation(
        x=q_range[1] * 0.7,
        y=half_spread * 1.2,
        text="Optimal Spread = (2/γ) * ln(1 + (γ/k))",
        showarrow=False,
        font=dict(size=12)
    )
    
    fig.add_annotation(
        x=q_range[0] * 0.7,
        y=reservation_adjustments[0] * 0.8,
        text="r(s,q,t) = s - q * γ * σ² * (T-t)",
        showarrow=False,
        font=dict(size=12)
    )
    
    return fig

# Helper function to update metrics based on parameter values
def update_metrics(gamma, sigma, k, T, A, q, t):
    """Calculate metrics based on current parameter values"""
    remaining_time = T - t
    res_price_adj = -q * gamma * (sigma**2) * remaining_time
    optimal_spread = gamma * (sigma**2) * remaining_time + (2/gamma) * np.log(1 + (gamma/k))
    bid_adj = res_price_adj - optimal_spread/2
    ask_adj = res_price_adj + optimal_spread/2
    bid_intensity = A * np.exp(-k * abs(bid_adj))
    ask_intensity = A * np.exp(-k * abs(ask_adj))
    
    return {
        'res_price_adj': res_price_adj,
        'optimal_spread': optimal_spread,
        'bid_adj': bid_adj,
        'ask_adj': ask_adj,
        'bid_intensity': bid_intensity,
        'ask_intensity': ask_intensity
    }

# Streamlit UI and App
def main():
    # Header
    st.title("Avellaneda-Stoikov Market Making Model")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["About", "Model Parameters", "Visualizations", "Run Simulation", "Strategy Comparison", "Documentation"])
    
    # Initialize parameters in session state if they don't exist
    if 'params' not in st.session_state:
        st.session_state.params = {
            'S0': 100.0,
            'T': 1.0,
            'sigma': 2.0,
            'M': 1000,
            'gamma': 0.1,
            'k': 1.5,
            'A': 140,
            'market_spread': 0.5,
            'num_sims': 100
        }
    
    # About section
    if page == "About":
        st.header("About the Avellaneda-Stoikov Model")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ## The Avellaneda-Stoikov Market Making Model
            
            The Avellaneda-Stoikov model is a mathematical framework for high-frequency market making in limit order book markets. 
            Developed by Marco Avellaneda and Sasha Stoikov (2008), it addresses the optimal strategy for submitting bid and ask orders 
            in electronic markets.
            
            ### Key Features:
            
            - **Inventory Risk Management**: The model accounts for the risk of holding inventory as price moves
            - **Order Arrival Model**: Uses Poisson processes with intensity dependent on quote depth
            - **Optimal Pricing Strategy**: Derives optimal bid/ask quotes based on reservation price and market conditions
            - **Utility Maximization**: Based on exponential utility to reflect risk aversion
            
            This application implements a simulation of the Avellaneda-Stoikov model, allowing you to explore how different
            parameters affect market making performance and compare various strategies.
            """)
        
        with col2:
            st.image("https://uci-seed-dataset.s3.ap-south-1.amazonaws.com/Orderbook.PNG", 
                    caption="Limit Order Book Visualization")
        
        with st.expander("More About Market Making"):
            st.markdown("""
            ### Market Making in Limit Order Books
            
            Market makers provide liquidity by continuously quoting bid and ask prices at which they're willing to buy and sell assets. 
            They profit from the bid-ask spread while managing inventory risk from price movements.
            
            In the Avellaneda-Stoikov framework:
            
            1. The **mid-price** follows a random walk (Brownian motion)
            2. The market maker's **reservation price** is their personal valuation adjusting for inventory
            3. **Order arrivals** follow a Poisson process with intensity depending on quote attractiveness
            4. The market maker aims to maximize expected terminal wealth utility
            
            The key insight is the **two-step approach**:
            - First, compute a personal reservation price based on current inventory
            - Second, set optimal bid/ask quotes around this price based on market order flow
            
            This application lets you simulate this model and visualize how different parameters 
            affect quoting strategy and performance.
            """)
    
    # Model Parameters section
    elif page == "Model Parameters":
        st.header("Model Parameters")
        
        tab1, tab2, tab3 = st.tabs(["Basic Parameters", "Advanced Parameters", "Market Parameters"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                st.session_state.params['S0'] = st.slider("Initial Mid-Price (S₀)", min_value=50.0, max_value=200.0, value=st.session_state.params['S0'], step=1.0,
                              help="The starting price of the asset")
                st.session_state.params['sigma'] = st.slider("Volatility (σ)", min_value=0.1, max_value=5.0, value=st.session_state.params['sigma'], step=0.1,
                                help="The volatility of the mid-price process")
            
            with col2:
                st.session_state.params['gamma'] = st.slider("Risk Aversion (γ)", min_value=0.01, max_value=1.0, value=st.session_state.params['gamma'], step=0.01,
                                help="Higher values mean more risk-averse behavior, reducing inventory exposure")
                st.session_state.params['k'] = st.slider("Order Intensity Shape (k)", min_value=0.5, max_value=3.0, value=st.session_state.params['k'], step=0.1,
                             help="Controls how quickly order execution probability decays with distance from mid-price")
        
        with tab2:
            col1, col2 = st.columns(2)
            
            with col1:
                st.session_state.params['T'] = st.slider("Time Horizon (T)", min_value=0.1, max_value=5.0, value=st.session_state.params['T'], step=0.1,
                             help="The time horizon for the simulation (in days)")
                st.session_state.params['M'] = st.slider("Number of Time Steps (M)", min_value=100, max_value=5000, value=st.session_state.params['M'], step=100,
                             help="More steps provide more granular simulation but take longer to run")
            
            with col2:
                st.session_state.params['num_sims'] = st.slider("Number of Simulations", min_value=10, max_value=500, value=st.session_state.params['num_sims'], step=10,
                                   help="More simulations provide more reliable statistics but take longer to run")
        
        with tab3:
            col1, col2 = st.columns(2)
            
            with col1:
                st.session_state.params['A'] = st.slider("Base Order Intensity (A)", min_value=50, max_value=300, value=st.session_state.params['A'], step=10,
                             help="Controls the overall rate of order arrivals")
            
            with col2:
                st.session_state.params['market_spread'] = st.slider("Market Spread (for Best Bid/Ask)", min_value=0.1, max_value=2.0, value=st.session_state.params['market_spread'], step=0.1,
                                        help="The spread used for the Best Bid/Ask benchmark strategy")
        
        # Parameter insights and formulas 
        with st.expander("Parameter Insights & Formulas"):
            st.markdown("""
            ### Key Formulas from Avellaneda-Stoikov Model
            
            #### Reservation Price
            The reservation price adjusts the mid-price based on current inventory:
            ```
            r(s,q,t) = s - q * γ * σ² * (T-t)
            ```
            
            #### Optimal Spread
            The optimal spread is given by:
            ```
            spread = γ * σ² * (T-t) + (2/γ) * ln(1 + (γ/k))
            ```
            
            #### Optimal Quotes
            The optimal bid and ask quotes are:
            ```
            bid = r(s,q,t) - spread/2
            ask = r(s,q,t) + spread/2
            ```
            
            #### Order Arrival Intensity
            The probability of order execution decays exponentially with distance from mid-price:
            ```
            λ(δ) = A * e^(-k*δ)
            ```
            
            ### Parameter Effects
            
            - **Higher γ (risk aversion)**: Narrows quotes around reservation price to reduce inventory
            - **Higher σ (volatility)**: Widens spreads to compensate for greater price risk
            - **Higher k**: Reduces spread as orders arrive more frequently at a given distance
            - **Higher A**: More orders overall, but doesn't affect optimal quotes
            """)
            
        # Show current optimal spread based on parameters
        st.subheader("Current Optimal Values")
        fixed_spread = (2/st.session_state.params['gamma']) * np.log(1 + (st.session_state.params['gamma']/st.session_state.params['k']))
        st.metric("Optimal Spread", f"{fixed_spread:.4f}")
    
    # Visualizations section
    elif page == "Visualizations":
        st.header("Model Visualizations")
        
        tab1, tab2, tab3 = st.tabs(["Pricing Structure", "Key Metrics", "Parameter Sensitivity"])
        
        with tab1:
            st.subheader("Reservation Price Surface")
            fig_reservation = plot_reservation_price_surface(
                gamma=st.session_state.params['gamma'], 
                sigma=st.session_state.params['sigma'], 
                T=st.session_state.params['T']
            )
            st.plotly_chart(fig_reservation, use_container_width=True)
            
            st.subheader("Optimal Quote Adjustments by Inventory Level")
            q_range_options = {
                "Small Range (-5 to 5)": (-5, 5),
                "Medium Range (-10 to 10)": (-10, 10),
                "Large Range (-20 to 20)": (-20, 20)
            }
            selected_range = st.selectbox("Inventory Range", options=list(q_range_options.keys()))
            q_range = q_range_options[selected_range]
            
            fig_quotes = plot_optimal_quotes_by_inventory(
                gamma=st.session_state.params['gamma'], 
                sigma=st.session_state.params['sigma'], 
                T=st.session_state.params['T'], 
                k=st.session_state.params['k'], 
                q_range=q_range
            )
            st.plotly_chart(fig_quotes, use_container_width=True)
            
            st.subheader("Order Arrival Intensity")
            delta_max = st.slider("Max Distance", min_value=1.0, max_value=10.0, value=5.0, step=0.5)
            
            fig_intensity = plot_order_arrival_intensity(
                k_values=[0.5, 1.0, st.session_state.params['k'], 2.0, 3.0],
                A=st.session_state.params['A'],
                delta_max=delta_max
            )
            st.plotly_chart(fig_intensity, use_container_width=True)
        
        with tab2:
            st.subheader("Key Market Making Metrics")
            st.write("""
            This section shows how key market making metrics like the Reservation Price, 
            Optimal Spread, and Order Intensity are calculated based on the model parameters.
            
            Adjust the parameters below to see how they affect these metrics.
            """)
            
            col1, col2 = st.columns(2)
            
            with col1:
                q_slider = st.slider("Inventory Position (q)", min_value=-10, max_value=10, value=0, step=1,
                                   help="The current inventory position of the market maker")
            
            with col2:
                t_slider = st.slider("Current Time (t)", min_value=0.0, max_value=1.0, value=0.0, step=0.05,
                                   help="The current time in the trading period [0,T]")
            
            # Calculate metrics
            metrics = update_metrics(
                st.session_state.params['gamma'], 
                st.session_state.params['sigma'], 
                st.session_state.params['k'], 
                st.session_state.params['T'], 
                st.session_state.params['A'], 
                q_slider, 
                t_slider
            )
            
            # Create metrics table
            metrics_data = {
                "Metric": [
                    "Reservation Price Adjustment",
                    "Optimal Spread",
                    "Bid Adjustment from Mid-Price",
                    "Ask Adjustment from Mid-Price",
                    "Bid Order Arrival Intensity",
                    "Ask Order Arrival Intensity"
                ],
                "Formula": [
                    "r(s,q,t) - s = -q * γ * σ² * (T-t)",
                    "γ * σ² * (T-t) + (2/γ) * ln(1 + (γ/k))",
                    "r(s,q,t) - s - spread/2",
                    "r(s,q,t) - s + spread/2",
                    "A * e^(-k*δᵇ)",
                    "A * e^(-k*δᵃ)"
                ],
                "Value": [
                    f"{metrics['res_price_adj']:.4f}",
                    f"{metrics['optimal_spread']:.4f}",
                    f"{metrics['bid_adj']:.4f}",
                    f"{metrics['ask_adj']:.4f}",
                    f"{metrics['bid_intensity']:.2f}",
                    f"{metrics['ask_intensity']:.2f}"
                ]
            }
            
            st.table(pd.DataFrame(metrics_data))
        
        with tab3:
            st.subheader("Parameter Sensitivity Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                gamma_min = st.number_input("Risk Aversion (γ) Min", min_value=0.01, max_value=1.0, value=0.05, step=0.01, format="%.2f")
                gamma_max = st.number_input("Risk Aversion (γ) Max", min_value=0.01, max_value=1.0, value=0.5, step=0.01, format="%.2f")
                gamma_steps = st.number_input("Risk Aversion Steps", min_value=2, max_value=10, value=5)
            
            with col2:
                k_min = st.number_input("Order Intensity Shape (k) Min", min_value=0.1, max_value=5.0, value=0.5, step=0.1, format="%.1f")
                k_max = st.number_input("Order Intensity Shape (k) Max", min_value=0.1, max_value=5.0, value=3.0, step=0.1, format="%.1f")
                k_steps = st.number_input("Order Intensity Shape Steps", min_value=2, max_value=10, value=5)
            
            # Generate heatmap button
            if st.button("Generate Heatmap Analysis"):
                with st.spinner("Running parameter sensitivity analysis..."):
                    # Generate parameter ranges
                    gamma_range = np.linspace(gamma_min, gamma_max, int(gamma_steps))
                    k_range = np.linspace(k_min, k_max, int(k_steps))
                    
                    # Generate heatmap
                    fig_heatmap = generate_heatmap_analysis(
                        gamma_values=gamma_range,
                        k_values=k_range,
                        S0=st.session_state.params['S0'],
                        T=st.session_state.params['T'],
                        sigma=st.session_state.params['sigma'],
                        M=st.session_state.params['M'],
                        A=st.session_state.params['A']
                    )
                    
                    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # Run Simulation section
    elif page == "Run Simulation":
        st.header("Run Simulation")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            run_single_button = st.button("Run Single Simulation", key="run_single")
        
        with col2:
            reset_button = st.button("Reset All", key="reset_sim")
        
        # Add tabs for results
        tab1, tab2 = st.tabs(["Simulation Results", "Key Metrics"])
        
        if run_single_button:
            with st.spinner("Running simulation..."):
                # Get parameters from session state
                params = {
                    'S0': st.session_state.params['S0'],
                    'T': st.session_state.params['T'],
                    'sigma': st.session_state.params['sigma'],
                    'M': int(st.session_state.params['M']),
                    'gamma': st.session_state.params['gamma'],
                    'k': st.session_state.params['k'],
                    'A': st.session_state.params['A'],
                    'seed': 42  # Fixed seed for reproducibility
                }
                
                # Check if parameters have changed
                if 'simulation_params' not in st.session_state or params != st.session_state.simulation_params:
                    st.session_state.simulation_params = params.copy()
                    
                    # Run simulation
                    model = AvellanedaStoikovModel(**params)
                    model.run_simulation()
                    results = model.get_results()
                    st.session_state.single_results = results
                    
                    # Generate plot
                    fig = model.generate_plotly_graph()
                    st.session_state.single_fig = fig
                
                # Display results in the tabs
                with tab1:
                    if 'single_fig' in st.session_state:
                        st.plotly_chart(st.session_state.single_fig, use_container_width=True)
                    else:
                        st.info("Run a simulation first to see results.")
                
                with tab2:
                    if 'single_results' in st.session_state:
                        results = st.session_state.single_results
                                    
                        # Display metrics in a nice format
                        col1, col2, col3, col4, col5 = st.columns(5)
                                    
                        with col1:
                            st.metric("Final Equity", f"{results['final_equity']:.2f}")
                                    
                        with col2:
                            st.metric("Avg Spread", f"{results['avg_spread']:.2f}")
                                    
                        with col3:
                            st.metric("Max Inventory", f"{int(results['max_inventory'])}")
                                    
                        with col4:
                            st.metric("Final Inventory", f"{int(results['final_inventory'])}")
                                    
                        with col5:
                            st.metric("Std Equity", f"{results['std_equity']:.2f}")
                    else:
                        st.info("No simulation results available. Run a simulation first.")
        
        if reset_button:
            # Reset session state
            if 'single_results' in st.session_state:
                del st.session_state.single_results
            if 'single_fig' in st.session_state:
                del st.session_state.single_fig
            if 'comparison_results' in st.session_state:
                del st.session_state.comparison_results
            if 'comparison_fig' in st.session_state:
                del st.session_state.comparison_fig
            if 'comparison_df' in st.session_state:
                del st.session_state.comparison_df
            
            st.success("All simulation results have been reset")

            if 'single_results' in st.session_state:
                results = st.session_state.single_results
            
            # Display metrics in a nice format
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("Final Equity", f"{results['final_equity']:.2f}")
            
            with col2:
                st.metric("Avg Spread", f"{results['avg_spread']:.2f}")
            
            with col3:
                st.metric("Max Inventory", f"{int(results['max_inventory'])}")
            
            with col4:
                st.metric("Final Inventory", f"{int(results['final_inventory'])}")
            
            with col5:
                st.metric("Std Equity", f"{results['std_equity']:.2f}")
        
        if reset_button:
            # Reset session state
            if 'single_results' in st.session_state:
                del st.session_state.single_results
            if 'single_fig' in st.session_state:
                del st.session_state.single_fig
            if 'comparison_results' in st.session_state:
                del st.session_state.comparison_results
            if 'comparison_fig' in st.session_state:
                del st.session_state.comparison_fig
            
            st.success("All simulation results have been reset")
    
    # Strategy Comparison section
    elif page == "Strategy Comparison":
        st.header("Strategy Comparison")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            run_comparison_button = st.button("Compare Strategies", key="run_comparison")
        
        # Add tabs for results
        tab1, tab2, tab3 = st.tabs(["Comparison Results", "Strategy Statistics", "Statistical Analysis"])
        
        if run_comparison_button:
            with st.spinner("Running strategy comparison..."):
                # Get parameters from session state
                params = {
                    'S0': st.session_state.params['S0'],
                    'T': st.session_state.params['T'],
                    'sigma': st.session_state.params['sigma'],
                    'M': int(st.session_state.params['M']),
                    'gamma': st.session_state.params['gamma'],
                    'k': st.session_state.params['k'],
                    'A': st.session_state.params['A'],
                    'num_sims': int(st.session_state.params['num_sims'])
                }
                
                # Check if parameters have changed
                if 'comparison_params' not in st.session_state or params != st.session_state.comparison_params:
                    st.session_state.comparison_params = params.copy()
                    
                    # Run comparison
                    strategy_results, fig_histograms, results_df = compare_strategies(**params)
                    
                    # Store results
                    st.session_state.comparison_results = strategy_results
                    st.session_state.comparison_fig = fig_histograms
                    st.session_state.comparison_df = results_df
                
                # Display results in the tabs
                with tab1:
                    st.plotly_chart(st.session_state.comparison_fig, use_container_width=True)
                
                with tab2:
                    # Display strategy comparison results
                    st.dataframe(st.session_state.comparison_results.reset_index())
                    
                    # Display optimal spread for each strategy
                    st.subheader("Strategy Configuration")
                    st.write(f"Optimal Spread for Inventory Strategy: {(2/st.session_state.params['gamma']) * np.log(1 + (st.session_state.params['gamma']/st.session_state.params['k'])):.4f}")
                    st.write(f"Market Spread for Best Bid/Ask Strategy: {st.session_state.params['market_spread']:.4f}")
                
                with tab3:
                    # Create statistical analysis
                    if 'comparison_df' in st.session_state:
                        results_df = st.session_state.comparison_df
                        
                        # Calculate Sharpe Ratios
                        sharpe_ratios = results_df.groupby('strategy').apply(
                            lambda x: x['final_equity'].mean() / x['final_equity'].std()
                        ).reset_index()
                        sharpe_ratios.columns = ['Strategy', 'Sharpe Ratio']
                        
                        # Create boxplot
                        fig_stats = go.Figure()
                        
                        for strategy in results_df['strategy'].unique():
                            strategy_data = results_df[results_df['strategy'] == strategy]
                            
                            fig_stats.add_trace(
                                go.Box(
                                    y=strategy_data['final_equity'],
                                    name=strategy,
                                    boxmean=True,
                                    jitter=0.3,
                                    pointpos=-1.8,
                                    boxpoints='all'
                                )
                            )
                        
                        fig_stats.update_layout(
                            title='Final Equity Distribution by Strategy',
                            yaxis_title='Final Equity',
                            template='plotly_white'
                        )
                        
                        st.plotly_chart(fig_stats, use_container_width=True, key=f"fig_stats_{int(time.time())}")
                        
                        st.subheader("Sharpe Ratios")
                        st.dataframe(sharpe_ratios)
                        
                        st.subheader("Strategy Insights")
                        st.markdown(f"""
                        - **Inventory Strategy**: Adjusts quotes based on current inventory position, leading to lower inventory risk
                        - **Symmetric Strategy**: Uses optimal spread but centered on mid-price rather than reservation price
                        - **Best Bid/Ask Strategy**: Uses fixed market spread centered on mid-price
                        
                        #### Key Findings
                        
                        - The Inventory strategy from Avellaneda-Stoikov model achieves better risk-adjusted returns
                        - The Symmetric strategy may generate higher absolute returns but with higher risk
                        - Risk aversion parameter (γ = {st.session_state.params['gamma']}) significantly impacts strategy performance
                        - Higher volatility (σ = {st.session_state.params['sigma']}) leads to wider optimal spreads
                        """)
                
                with tab2:
                    # Display strategy comparison results
                    st.dataframe(st.session_state.comparison_results.reset_index())
                    
                    # Display optimal spread for each strategy
                    st.subheader("Strategy Configuration")
                    gamma = st.session_state.params['gamma']
                    k = st.session_state.params['k']
                    st.write(f"Optimal Spread for Inventory Strategy: {(2/gamma) * np.log(1 + (gamma/k)):.4f}")
                    st.write(f"Market Spread for Best Bid/Ask Strategy: {0.5:.4f}")
                
                with tab3:
                    # Create statistical analysis
                    if 'comparison_df' in st.session_state:
                        results_df = st.session_state.comparison_df
                        sigma = st.session_state.params['sigma']
                        
                        # Calculate Sharpe Ratios
                        sharpe_ratios = results_df.groupby('strategy').apply(
                            lambda x: x['final_equity'].mean() / x['final_equity'].std()
                        ).reset_index()
                        sharpe_ratios.columns = ['Strategy', 'Sharpe Ratio']
                        
                        # Create boxplot
                        fig_stats = go.Figure()
                        
                        for strategy in results_df['strategy'].unique():
                            strategy_data = results_df[results_df['strategy'] == strategy]
                            
                            fig_stats.add_trace(
                                go.Box(
                                    y=strategy_data['final_equity'],
                                    name=strategy,
                                    boxmean=True,
                                    jitter=0.3,
                                    pointpos=-1.8,
                                    boxpoints='all'
                                )
                            )
                        
                        fig_stats.update_layout(
                            title='Final Equity Distribution by Strategy',
                            yaxis_title='Final Equity',
                            template='plotly_white'
                        )
                        
                        st.plotly_chart(fig_stats, use_container_width=True, key=f"fig_stats_{int(time.time())}")
                        
                        st.subheader("Sharpe Ratios")
                        st.dataframe(sharpe_ratios)
                        
                        st.subheader("Strategy Insights")
                        st.markdown(f"""
                        - **Inventory Strategy**: Adjusts quotes based on current inventory position, leading to lower inventory risk
                        - **Symmetric Strategy**: Uses optimal spread but centered on mid-price rather than reservation price
                        - **Best Bid/Ask Strategy**: Uses fixed market spread centered on mid-price
                        
                        #### Key Findings
                        
                        - The Inventory strategy from Avellaneda-Stoikov model achieves better risk-adjusted returns
                        - The Symmetric strategy may generate higher absolute returns but with higher risk
                        - Risk aversion parameter (γ = {gamma}) significantly impacts strategy performance
                        - Higher volatility (σ = {sigma}) leads to wider optimal spreads
                        """)
    
    # Documentation section
    elif page == "Documentation":
        st.header("Documentation & Deployment")
        
        tab1, tab2, tab3 = st.tabs(["Theoretical Background", "Deployment Instructions", "Course Alignment"])
        
        with tab1:
            st.markdown("""
            ## Theoretical Background
            
            ### The Avellaneda-Stoikov Framework
            
            The Avellaneda-Stoikov model (2008) provides a mathematical framework for market making in electronic limit order books. 
            The core problem it addresses is: *How should a market maker optimally place bid and ask quotes to maximize expected utility while managing inventory risk?*
            
            #### Key Components of the Model:
            
            1. **Mid-Price Evolution**: The stock's mid-price follows a Brownian motion
               ```
               dS_t = σdW_t
               ```
            
            2. **Inventory Risk**: The market maker faces risk from holding non-zero inventory as the price moves
            
            3. **Reservation Price**: The market maker's personal valuation of the stock, adjusted for current inventory
               ```
               r(s,q,t) = s - q * γ * σ² * (T-t)
               ```
            
            4. **Order Arrivals**: Buy and sell orders arrive as Poisson processes with intensity dependent on quote attractiveness
               ```
               λ(δ) = A * e^(-k*δ)
               ```
               where δ is the distance from mid-price
            
            5. **Utility Maximization**: The market maker aims to maximize expected exponential utility of terminal wealth
            
            #### The Two-Step Solution:
            
            1. First, compute a personal reservation price given current inventory
            2. Second, calibrate the bid and ask quotes to market conditions
            
            #### Benefits of the Model:
            
            - **Inventory Management**: Automatically adjusts quotes to reduce inventory
            - **Risk Management**: Accounts for price volatility and time horizon
            - **Optimal Spread**: Derives spread based on risk aversion and market order flow
            
            ### Extensions and Modifications
            
            The original model has been extended in numerous ways:
            
            - Incorporating adverse selection risk
            - Adding multiple trading venues
            - Considering market impact effects
            - Accounting for limit order book depth
            
            This simulation implements the core Avellaneda-Stoikov model with additional features for strategy comparison and parameter analysis.
            """)
        
        with tab2:
            st.markdown("""
            ## Deployment Instructions
            
            ### Local Deployment
            
            #### Prerequisites:
            - Python 3.8 or higher
            - pip package manager
            
            #### Step 1: Clone or download the project
            ```bash
            git clone <repository-url>
            cd avellaneda-stoikov-simulator
            ```
            
            #### Step 2: Install dependencies
            ```bash
            pip install -r requirements.txt
            ```
            
            #### Step 3: Run the application
            ```bash
            streamlit run app.py
            ```
            
            The application will be available at `http://localhost:8501`.
            
            ### Cloud Deployment
            
            #### Deploying on Streamlit Cloud
            
            1. Push your code to a GitHub repository
            2. Sign up for Streamlit Cloud at https://streamlit.io/cloud
            3. Create a new app and connect it to your GitHub repository
            4. Select the app.py file as the main file
            5. Deploy
            
            #### Deploying on Heroku
            
            1. Create a `Procfile` with the following content:
               ```
               web: sh setup.sh && streamlit run app.py
               ```
               
            2. Create a `setup.sh` file:
               ```bash
               mkdir -p ~/.streamlit/
               
               echo "\
               [server]\n\
               headless = true\n\
               port = $PORT\n\
               enableCORS = false\n\
               " > ~/.streamlit/config.toml
               ```
               
            3. Create a `requirements.txt` file with all dependencies
            4. Deploy to Heroku:
               ```bash
               heroku create
               git push heroku main
               ```
            
            ### Docker Deployment
            
            #### Step 1: Create a Dockerfile
            ```dockerfile
            FROM python:3.9-slim
            
            WORKDIR /app
            
            COPY requirements.txt .
            RUN pip install --no-cache-dir -r requirements.txt
            
            COPY . .
            
            EXPOSE 8501
            
            ENTRYPOINT ["streamlit", "run"]
            CMD ["app.py"]
            ```
            
            #### Step 2: Build and run the Docker image
            ```bash
            docker build -t avellaneda-stoikov .
            docker run -p 8501:8501 avellaneda-stoikov
            ```
            
            ### Project Structure
            
            ```
            avellaneda-stoikov-simulator/
            ├── app.py               # Main application code
            ├── requirements.txt     # Dependencies
            ├── Dockerfile           # For Docker deployment
            ├── setup.sh             # For Heroku deployment
            ├── Procfile             # For Heroku deployment
            └── README.md            # Project documentation
            ```
            
            ### File: requirements.txt
            
            ```
            streamlit>=1.18.0
            numpy>=1.20.0
            pandas>=1.3.0
            plotly>=5.10.0
            matplotlib>=3.5.0
            ```
            """)
        
        with tab3:
            st.markdown("""
            ## Course Alignment: Introduction to Simulation & Modeling
            
            This application aligns with the CSE3171 "Introduction to Simulation & Modeling" course by implementing
            key concepts covered in the curriculum:
            
            ### Unit I - Introduction to Simulation
            
            - **Definition of system and simulation**: The Avellaneda-Stoikov model simulates a market making system
            - **Modeling a continuous system**: Implements a stochastic continuous-time stock price model
            - **Simulation methodology**: Follows proper simulation steps from model definition to analysis
            
            ### Unit II - Random Numbers
            
            - **Role of random numbers in simulation**: Uses pseudo-random number generation for Brownian motion
            - **Random variates generation**: Implements normal and exponential random variable generation
            
            ### Unit III - Simulation Techniques
            
            - **Monte Carlo methods**: Uses Monte Carlo simulation to analyze strategy performance
            - **Simulation of ODE/SDE**: Implements Euler-Maruyama method to simulate stochastic differential equations
              ```python
              # Euler-Maruyama implementation in the code:
              self.S[t] = self.S[t-1] + self.sigma * self.sqrt_dt * z
              ```
            
            ### Unit IV - Analysis of Simulation Data
            
            - **Data collection and analysis**: Gathers statistics across multiple simulations
            - **Parameter estimation**: Estimates optimal parameters through sensitivity analysis
            - **Verification and validation**: Compares model outcomes with theoretical predictions
            
            ### Unit V - Queuing Models & Stochastic Systems
            
            - **Queuing theory concepts**: Models order arrivals as Poisson processes with state-dependent intensities
            - **Markov chain applications**: Inventory process follows Markov property 
            - **Social/biological models**: Implements diffusion models similar to Bass diffusion
            
            ### Course Outcomes Addressed:
            
            - **CO-1**: Understanding random variables and processes (mid-price as Brownian motion)
            - **CO-2**: Applying simulation techniques to study system behavior (market making strategy)
            - **CO-3**: Implementing methods to simulate SDEs (Euler-Maruyama method)
            - **CO-4**: Demonstrating applications of simulation (optimal quoting strategy)
            - **CO-5**: Analyzing and evaluating the performance of stochastic systems
            - **CO-6**: Creating solutions for real-world financial systems
            """)

# Run the app
if __name__ == "__main__":
    main()
