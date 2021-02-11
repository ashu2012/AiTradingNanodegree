
# Advanced Portfolio Optimization using cvxpy

## Install cvxpy and other libraries


```python
import sys
!{sys.executable} -m pip install -r requirements.txt
```

    Requirement already satisfied: colour==0.1.5 in /opt/conda/lib/python3.6/site-packages (from -r requirements.txt (line 1)) (0.1.5)
    Requirement already satisfied: cvxpy==1.0.3 in /opt/conda/lib/python3.6/site-packages (from -r requirements.txt (line 2)) (1.0.3)
    Requirement already satisfied: cycler==0.10.0 in /opt/conda/lib/python3.6/site-packages/cycler-0.10.0-py3.6.egg (from -r requirements.txt (line 3)) (0.10.0)
    Requirement already satisfied: numpy==1.14.5 in /opt/conda/lib/python3.6/site-packages (from -r requirements.txt (line 4)) (1.14.5)
    Requirement already satisfied: pandas==0.21.1 in /opt/conda/lib/python3.6/site-packages (from -r requirements.txt (line 5)) (0.21.1)
    Requirement already satisfied: plotly==2.2.3 in /opt/conda/lib/python3.6/site-packages (from -r requirements.txt (line 6)) (2.2.3)
    Requirement already satisfied: pyparsing==2.2.0 in /opt/conda/lib/python3.6/site-packages (from -r requirements.txt (line 7)) (2.2.0)
    Requirement already satisfied: python-dateutil==2.6.1 in /opt/conda/lib/python3.6/site-packages (from -r requirements.txt (line 8)) (2.6.1)
    Requirement already satisfied: pytz==2017.3 in /opt/conda/lib/python3.6/site-packages (from -r requirements.txt (line 9)) (2017.3)
    Requirement already satisfied: requests==2.18.4 in /opt/conda/lib/python3.6/site-packages (from -r requirements.txt (line 10)) (2.18.4)
    Requirement already satisfied: scipy==1.0.0 in /opt/conda/lib/python3.6/site-packages (from -r requirements.txt (line 11)) (1.0.0)
    Requirement already satisfied: scikit-learn==0.19.1 in /opt/conda/lib/python3.6/site-packages (from -r requirements.txt (line 12)) (0.19.1)
    Requirement already satisfied: six==1.11.0 in /opt/conda/lib/python3.6/site-packages (from -r requirements.txt (line 13)) (1.11.0)
    Requirement already satisfied: tqdm==4.19.5 in /opt/conda/lib/python3.6/site-packages (from -r requirements.txt (line 14)) (4.19.5)
    Requirement already satisfied: ecos>=2 in /opt/conda/lib/python3.6/site-packages (from cvxpy==1.0.3->-r requirements.txt (line 2)) (2.0.7.post1)
    Requirement already satisfied: toolz in /opt/conda/lib/python3.6/site-packages (from cvxpy==1.0.3->-r requirements.txt (line 2)) (0.8.2)
    Requirement already satisfied: fastcache in /opt/conda/lib/python3.6/site-packages (from cvxpy==1.0.3->-r requirements.txt (line 2)) (1.0.2)
    Requirement already satisfied: scs>=1.1.3 in /opt/conda/lib/python3.6/site-packages (from cvxpy==1.0.3->-r requirements.txt (line 2)) (2.1.2)
    Requirement already satisfied: multiprocess in /opt/conda/lib/python3.6/site-packages (from cvxpy==1.0.3->-r requirements.txt (line 2)) (0.70.11.1)
    Requirement already satisfied: osqp in /opt/conda/lib/python3.6/site-packages (from cvxpy==1.0.3->-r requirements.txt (line 2)) (0.6.2.post0)
    Requirement already satisfied: nbformat>=4.2 in /opt/conda/lib/python3.6/site-packages (from plotly==2.2.3->-r requirements.txt (line 6)) (4.4.0)
    Requirement already satisfied: decorator>=4.0.6 in /opt/conda/lib/python3.6/site-packages (from plotly==2.2.3->-r requirements.txt (line 6)) (4.0.11)
    Requirement already satisfied: chardet<3.1.0,>=3.0.2 in /opt/conda/lib/python3.6/site-packages (from requests==2.18.4->-r requirements.txt (line 10)) (3.0.4)
    Requirement already satisfied: idna<2.7,>=2.5 in /opt/conda/lib/python3.6/site-packages (from requests==2.18.4->-r requirements.txt (line 10)) (2.6)
    Requirement already satisfied: urllib3<1.23,>=1.21.1 in /opt/conda/lib/python3.6/site-packages (from requests==2.18.4->-r requirements.txt (line 10)) (1.22)
    Requirement already satisfied: certifi>=2017.4.17 in /opt/conda/lib/python3.6/site-packages (from requests==2.18.4->-r requirements.txt (line 10)) (2019.11.28)
    Requirement already satisfied: dill>=0.3.3 in /opt/conda/lib/python3.6/site-packages (from multiprocess->cvxpy==1.0.3->-r requirements.txt (line 2)) (0.3.3)
    Requirement already satisfied: qdldl in /opt/conda/lib/python3.6/site-packages (from osqp->cvxpy==1.0.3->-r requirements.txt (line 2)) (0.1.5.post0)
    Requirement already satisfied: jupyter-core in /opt/conda/lib/python3.6/site-packages (from nbformat>=4.2->plotly==2.2.3->-r requirements.txt (line 6)) (4.4.0)
    Requirement already satisfied: jsonschema!=2.5.0,>=2.4 in /opt/conda/lib/python3.6/site-packages (from nbformat>=4.2->plotly==2.2.3->-r requirements.txt (line 6)) (2.6.0)
    Requirement already satisfied: ipython-genutils in /opt/conda/lib/python3.6/site-packages (from nbformat>=4.2->plotly==2.2.3->-r requirements.txt (line 6)) (0.2.0)
    Requirement already satisfied: traitlets>=4.1 in /opt/conda/lib/python3.6/site-packages (from nbformat>=4.2->plotly==2.2.3->-r requirements.txt (line 6)) (4.3.2)


## Imports


```python
import cvxpy as cvx
import numpy as np
import quiz_tests_advanced
```

## What's our objective?
http://www.cvxpy.org/

Let's see how we can use optimization to meet a more advanced objective.  We want to both minimize the portfolio variance and also want to closely track a market cap weighted index.  In other words, we're trying to minimize the distance between the weights of our portfolio and the weights of the index.

$Minimize \left [ \sigma^2_p + \lambda \sqrt{\sum_{1}^{m}(weight_i - indexWeight_i)^2} \right  ]$ where $m$ is the number of stocks in the portfolio, and $\lambda$ is a scaling factor that you can choose.


## Hints

### x vector
To create a vector of M variables $\mathbf{x} = \begin{bmatrix}
x_1 &...& x_M
\end{bmatrix}
$
we can use `cvx.Variable(m)`

### covariance matrix
If we have $m$ stock series, the covariance matrix is an $m \times m$ matrix containing the covariance between each pair of stocks.  We can use [numpy.cov](https://docs.scipy.org/doc/numpy/reference/generated/numpy.cov.html) to get the covariance.  We give it a 2D array in which each row is a stock series, and each column is an observation at the same period of time.

The covariance matrix $\mathbf{P} = 
\begin{bmatrix}
\sigma^2_{1,1} & ... & \sigma^2_{1,m} \\ 
... & ... & ...\\
\sigma_{m,1} & ... & \sigma^2_{m,m}  \\
\end{bmatrix}$

### portfolio variance
We can write the portfolio variance $\sigma^2_p = \mathbf{x^T} \mathbf{P} \mathbf{x}$

Recall that the $\mathbf{x^T} \mathbf{P} \mathbf{x}$ is called the quadratic form.
We can use the cvxpy function `quad_form(x,P)` to get the quadratic form.

### Distance from index weights
We want portfolio weights that track the index closely.  So we want to minimize the distance between them.
Recall from the Pythagorean theorem that you can get the distance between two points in an x,y plane by adding the square of the x and y distances and taking the square root.  Extending this to any number of dimensions is called the L2 norm.  So: $\sqrt{\sum_{1}^{n}(weight_i - indexWeight_i)^2}$  Can also be written as $\left \| \mathbf{x} - \mathbf{index} \right \|_2$.  There's a cvxpy function called [norm()](https://www.cvxpy.org/api_reference/cvxpy.atoms.other_atoms.html#norm)
`norm(x, p=2, axis=None)`.  The default is already set to find an L2 norm, so you would pass in one argument, which is the difference between your portfolio weights and the index weights.

### objective function
We want to minimize both the portfolio variance and the distance of the portfolio weights from the index weights.
We also want to choose a `scale` constant, which is $\lambda$ in the expression. This lets us choose how much priority we give to minimizing the difference from the index, relative to minimizing the variance of the portfolio.  If you choose a higher value for `scale` ($\lambda$), do you think this gives more priority to minimizing the difference, or minimizing the variance?

We can find the objective function using cvxpy `objective = cvx.Minimize()`.  Can you guess what to pass into this function?

### constraints
We can also define our constraints in a list.  For example, you'd want the weights to sum to one. So $\sum_{1}^{n}x = 1$.  You may also need to go long only, which means no shorting, so no negative weights.  So $x_i >0 $ for all $i$. you could save a variable as `[x >= 0, sum(x) == 1]`, where x was created using `cvx.Variable()`.

### optimization
So now that we have our objective function and constraints, we can solve for the values of $\mathbf{x}$.
cvxpy has the constructor `Problem(objective, constraints)`, which returns a `Problem` object.

The `Problem` object has a function solve(), which returns the minimum of the solution.  In this case, this is the minimum variance of the portfolio.

It also updates the vector $\mathbf{x}$.

We can check out the values of $x_A$ and $x_B$ that gave the minimum portfolio variance by using `x.value`

## Quiz


```python
import cvxpy as cvx
import numpy as np

def optimize_portfolio(returns, index_weights, scale=.00001):
    """
    Create a function that takes the return series of a set of stocks, the index weights,
    and scaling factor. The function will minimize a combination of the portfolio variance
    and the distance of its weights from the index weights.  
    The optimization will be constrained to be long only, and the weights should sum to one.
    
    Parameters
    ----------
    returns : numpy.ndarray
        2D array containing stock return series in each row.
        
    index_weights : numpy.ndarray
        1D numpy array containing weights of the index.
        
    scale : float
        The scaling factor applied to the distance between portfolio and index weights
        
    Returns
    -------
    x : np.ndarray
        A numpy ndarray containing the weights of the stocks in the optimized portfolio
    """
    # TODO: Use cvxpy to determine the weights on the assets
    # that minimizes the combination of portfolio variance and distance from index weights
    
    # number of stocks m is number of rows of returns, and also number of index weights
    #m = 
    
    #covariance matrix of returns
    #cov = 
    
    # x variables (to be found with optimization)
    #x = 
    
    #portfolio variance, in quadratic form
    #portfolio_variance = 
    
    # euclidean distance (L2 norm) between portfolio and index weights
    #distance_to_index = 
    
    #objective function
    #objective = 
    
    #constraints
    #constraints = 

    #use cvxpy to solve the objective
    
    #retrieve the weights of the optimized portfolio
    #x_values
    m = returns.shape[0]
    cov = np.cov(returns)
    x = cvx.Variable(m)
    portfolio_variance = cvx.quad_form(x, cov)
    distance_to_index = cvx.norm(x - index_weights)
    objective = cvx.Minimize(portfolio_variance + scale * distance_to_index)
    constraints = [x >= 0, sum(x) == 1]
    cvx.Problem(objective, constraints).solve()
    x_values = x.value
    return x_values
    

quiz_tests_advanced.test_optimize_portfolio(optimize_portfolio)
```

    Tests Passed



```python
"""Test with a 3 simulated stock return series"""
days_per_year = 252
years = 3
total_days = days_per_year * years

return_market = np.random.normal(loc=0.05, scale=0.3, size=days_per_year)
return_1 = np.random.uniform(low=-0.000001, high=.000001, size=days_per_year) + return_market
return_2 = np.random.uniform(low=-0.000001, high=.000001, size=days_per_year) + return_market
return_3 = np.random.uniform(low=-0.000001, high=.000001, size=days_per_year) + return_market
returns = np.array([return_1, return_2, return_3])

"""simulate index weights"""
index_weights = np.array([0.9,0.15,0.05])

"""try out your optimization function"""
x_values = optimize_portfolio(returns, index_weights, scale=.00001)

print(f"The optimized weights are {x_values}, which sum to {sum(x_values):.2f}")
```

    The optimized weights are [0.86738322 0.11642248 0.01619431], which sum to 1.00


If you're feeling stuck, you can check out the solution [here](m3l4_cvxpy_advanced_solution.ipynb)


```python

```
