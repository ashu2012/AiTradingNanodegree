
# Covariance Matrix

## Install libraries


```python
import sys
!{sys.executable} -m pip install -r requirements.txt
```

    Collecting numpy==1.14.5 (from -r requirements.txt (line 1))
    [?25l  Downloading https://files.pythonhosted.org/packages/68/1e/116ad560de97694e2d0c1843a7a0075cc9f49e922454d32f49a80eb6f1f2/numpy-1.14.5-cp36-cp36m-manylinux1_x86_64.whl (12.2MB)
    [K    100% |████████████████████████████████| 12.2MB 2.9MB/s eta 0:00:01  4% |█▋                              | 593kB 16.1MB/s eta 0:00:01    45% |██████████████▋                 | 5.5MB 19.0MB/s eta 0:00:01    53% |█████████████████               | 6.5MB 19.1MB/s eta 0:00:01    61% |███████████████████▊            | 7.5MB 21.2MB/s eta 0:00:01    69% |██████████████████████▍         | 8.5MB 21.6MB/s eta 0:00:01    95% |██████████████████████████████▌ | 11.6MB 22.2MB/s eta 0:00:01
    [31mtensorflow 1.3.0 requires tensorflow-tensorboard<0.2.0,>=0.1.0, which is not installed.[0m
    [?25hInstalling collected packages: numpy
      Found existing installation: numpy 1.12.1
        Uninstalling numpy-1.12.1:
          Successfully uninstalled numpy-1.12.1
    Successfully installed numpy-1.14.5


## Imports


```python
import numpy as np
import quiz_tests
```

## Hints

### covariance matrix
If we have $m$ stock series, the covariance matrix is an $m \times m$ matrix containing the covariance between each pair of stocks.  We can use [numpy.cov](https://docs.scipy.org/doc/numpy/reference/generated/numpy.cov.html) to get the covariance.  We give it a 2D array in which each row is a stock series, and each column is an observation at the same period of time.

The covariance matrix $\mathbf{P} = 
\begin{bmatrix}
\sigma^2_{1,1} & ... & \sigma^2_{1,m} \\ 
... & ... & ...\\
\sigma_{m,1} & ... & \sigma^2_{m,m}  \\
\end{bmatrix}$

## Quiz


```python
import numpy as np

def covariance_matrix(returns):
    """
    Create a function that takes the return series of a set of stocks
    and calculates the covariance matrix.
    
    Parameters
    ----------
    returns : numpy.ndarray
        2D array containing stock return series in each row.
                
    Returns
    -------
    x : np.ndarray
        A numpy ndarray containing the covariance matrix
    """
    
    #covariance matrix of returns
    #cov = 
    #print(returns)   
    cov = np.cov(returns) 

    return cov

quiz_tests.test_covariance_matrix(covariance_matrix)
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

"""try out your function"""
cov = covariance_matrix(returns)

print(f"The covariance matrix is \n{cov}")
```

    The covariance matrix is 
    [[0.09158481 0.0915848  0.09158482]
     [0.0915848  0.09158479 0.09158482]
     [0.09158482 0.09158482 0.09158484]]


If you're stuck, you can also check out the solution [here](m3l4_covariance_solution.ipynb)


```python

```
