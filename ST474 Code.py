#!/usr/bin/env python
# coding: utf-8

# In[70]:


import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as sk
import statsmodels.api as sm
from datetime import datetime as dt
from dateutil.relativedelta import relativedelta as rd
import schedule


# In[105]:


from functools import wraps
from time import time

def timing(f):
    @wraps(f)
    def wrap(*args,**kw):
        tstart = time()
        result = f(*args,**kw)
        tend = time()
        print('func:%r args: [%r,%r] took: %2.4f sec' %               (f.__name__,args,kw,tend-tstart))
        return result
    return wrap
        
        
    


# In[164]:


#Inputs
r = 0.06 # Risk-free rate taken from US treasury bonds 
sigma = 0.2
seed = None
S0 = 100 #Initial Stock Price
K = 100  #Strike Price
T = 1    #Time to Maturity in Years
N = 3    #Number of Time Steps
u = 1.1  #up-factor in binomial model
d = 1/u  #down-factor in binomial model
optiontype = 'P' #Choose 'C' for call & 'P' for Put


# In[157]:


def gbm_stock_price(S0, r, sigma, T, N, seed=None):
    np.random.seed(seed)
    dt = T / N
    t = np.linspace(0, T, N+1)
    W = np.random.standard_normal(size=N+1) 
    W = np.cumsum(W) * np.sqrt(dt)  # Brownian motion
    X = (r - 0.5 * sigma**2) * t + sigma * W  # GBM process
    S = S0 * np.exp(X)  # Stock price process
    S[0] = S0
    return S


# In[158]:


# Function to plot stock price paths
def plot_stock_price_paths(S0, r, sigma, T, N, num_paths=5, seed=None):
    plt.figure(figsize=(10, 6))
    
    for i in range(num_paths):
        stock_prices = gbm_stock_price(S0, r, sigma, T, N, seed)
        plt.plot(np.linspace(0, T, N+1), stock_prices, linewidth=2)

    plt.title('Stock Price Paths (GBM)')
    plt.xlabel('Time to Maturity (Years)')
    plt.ylabel('Stock Price')
    plt.grid(True)
    plt.show()


# In[161]:


plot_stock_price_paths(S0, r, sigma, T, N, num_paths=10, seed=seed)


# In[165]:


@timing
def american_slow_tree(K,T,S0,r,N,u,d,opttype='P'):
    dt = T/N
    p = (np.exp(r*dt)-d)/(u-d)
    discount = np.exp(-r*dt)
    
    stock_prices = np.zeros(N+1)
    for i in range(0,N+1):
        stock_prices[i] = S0 *u**i*d**(N-i)
    
    payoff = np.zeros(N+1)
    for i in range(0,N+1):
        if optiontype == 'P':
            payoff[i] = max(0,K-stock_prices[i])
        else:
            payoff[i] = max(0,stock_prices[i]-k)
    
    for i in np.arange(N-1,-1,-1):
        for j in range(0,i+1):
            stock_prices = S0 *u**j*d**(i-j)
            payoff[j] = discount* (p*payoff[j+1]+(1-p)*payoff[j])
            if optiontype == 'P':
                payoff[j] = max(payoff[j],K-stock_prices)
            else:
                payoff[j] = max(payoff[j],stock_prices-k)
    return payoff[0]
            
american_slow_tree(K,T,S0,r,N,u,d,opttype='P')  


# In[166]:


@timing
def american_fast_tree(K,T,S0,r,N,u,d,opttype='P'):
    dt = T/N
    p = (np.exp(r*dt)-d)/(u-d)
    discount = np.exp(-r*dt)  
    
    stock_price = S0*d**(np.arange(N,-1,-1))*u**(np.arange(0,N+1,1))
    
    if optiontype == 'P':
        payoff = np.maximum(0,K-stock_price)
    else:
        payoff = np.maximum(0,stock_price-k)
    for i in np.arange(N-1,-1,-1):
        stock_price = S0*d**(np.arange(i,-1,-1))*u**(np.arange(0,i+1,1))
        payoff[:i+1] = discount * (p*payoff[1:i+2]+(1-p)*payoff[0:i+1])
        payoff = payoff[:-1]
        if optiontype == 'P':
            payoff = np.maximum(payoff,K-stock_price)
        else:
            payoff = np.maximum(payoff,stock_price-k)
    return payoff[0]

            
american_fast_tree(K,T,S0,r,N,u,d,opttype='P') 


# In[168]:


@timing
def american_slow_tree(K,T,S0,r,N,u,d,opttype='P'):
    dt = T/N
    p = (np.exp(r*dt)-d)/(u-d)
    discount = np.exp(-r*dt)
    
    stock_prices = gbm_stock_price(S0, r, sigma, T, N, seed)
    
    payoff = np.zeros(N+1)
    for i in range(0,N+1):
        if optiontype == 'P':
            payoff[i] = max(0,K-stock_prices[i])
        else:
            payoff[i] = max(0,stock_prices[i]-k)
    
    for i in np.arange(N-1,-1,-1):
        for j in range(0,i+1):
            stock_prices = S0 *u**j*d**(i-j)
            payoff[j] = discount* (p*payoff[j+1]+(1-p)*payoff[j])
            if optiontype == 'P':
                payoff[j] = max(payoff[j],K-stock_prices)
            else:
                payoff[j] = max(payoff[j],stock_prices-k)
    return payoff[0]
            
american_slow_tree(K,T,S0,r,N,u,d,opttype='P') 


# In[169]:


@timing
def american_fast_tree(K,T,S0,r,N,u,d,opttype='P'):
    dt = T/N
    p = (np.exp(r*dt)-d)/(u-d)
    discount = np.exp(-r*dt)  
    
    stock_price = gbm_stock_price(S0, r, sigma, T, N, seed)
    
    if optiontype == 'P':
        payoff = np.maximum(0,K-stock_price)
    else:
        payoff = np.maximum(0,stock_price-k)
    for i in np.arange(N-1,-1,-1):
        stock_price = S0*d**(np.arange(i,-1,-1))*u**(np.arange(0,i+1,1))
        payoff[:i+1] = discount * (p*payoff[1:i+2]+(1-p)*payoff[0:i+1])
        payoff = payoff[:-1]
        if optiontype == 'P':
            payoff = np.maximum(payoff,K-stock_price)
        else:
            payoff = np.maximum(payoff,stock_price-k)
    return payoff[0]

            
american_fast_tree(K,T,S0,r,N,u,d,opttype='P') 


# In[147]:


for N in [3,10,25,100, 500, 1000]:
    fast = american_fast_tree(K,T,S0,r,N,u,d,opttype='P') 
    slow = american_slow_tree(K,T,S0,r,N,u,d,opttype='P') 
    


# In[ ]:


@timing
def american_trinomial_tree_vectorized(K, T, S0, r, N, u, d, sigma, opttype='P', seed=None):
    dt = T / N
    M = 2 * N + 1  # Number of nodes in each time step for a trinomial tree
    
    # Generate stock price paths using a trinomial tree
    stock_price = gbm_stock_price(S0, r, sigma, T, N, seed)

    # Calculate option values at maturity
    if optiontype == 'P':
        payoff = np.maximum(0,K-stock_price)
    else:
        payoff = np.maximum(0,stock_price-k)

    # Backward induction
    for i in np.arange(N - 1, -1, -1):
        stock_prices = S0 * u**(np.arange(-i, i+1, 2) // 2) * d**(i - np.arange(-i, i+1, 2) // 2)
        option_values = np.exp(-r * dt) * (
            0.5 * (np.roll(payoff, 2) + np.roll(payoff, -2)) +
            0.5 * (2 * payoff + np.roll(payoff, -1) + np.roll(payoff, 1)) +
            0.5 * (payoff + np.roll(payoff, 1) + np.roll(payoff, -1)) -
            0.5 * (payoff + np.roll(payoff, -2) + np.roll(payoff, 2))
        )

        if opttype == 'P':
            option_values = np.maximum(option_values, K - stock_prices)
        else:
            option_values = np.maximum(option_values, stock_prices - K)

    return option_values[0]

# Example usage:
S0 = 100
r = 0.05
sigma = 0.2
T = 1
N = 100
u = np.exp(sigma * np.sqrt(2 * T / N))
d = 1 / u
seed = 42

result_trinomial_vectorized = american_trinomial_tree_vectorized(100, 1, 100, 0.05, 100, u, d, sigma, opttype='P', seed=seed)
print(f"Trinomial (Vectorized) Result: {result_trinomial_vectorized}")


# In[ ]:




