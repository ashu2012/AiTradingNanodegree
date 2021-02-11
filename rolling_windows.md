
# Rolling Windows

## Pandas.DataFrame.rolling


You've just learned about rolling windows.  Let's see how we can use rolling function in pandas to create the rolling windows

First, let's create a simple dataframe!



```python
import numpy as np
import pandas as pd
from datetime import datetime

dates = pd.date_range(datetime.strptime('10/10/2018', '%m/%d/%Y'), periods=11, freq='D')
close_prices = np.arange(len(dates))

close = pd.Series(close_prices, dates)
close
```




    2018-10-10     0
    2018-10-11     1
    2018-10-12     2
    2018-10-13     3
    2018-10-14     4
    2018-10-15     5
    2018-10-16     6
    2018-10-17     7
    2018-10-18     8
    2018-10-19     9
    2018-10-20    10
    Freq: D, dtype: int64



Here, we will introduce rolling function from pandas.  The rolling function helps to provide rolling windows that can be customized through different parameters.  

You can learn more about [rolling function here](https://pandas.pydata.org/pandas-docs/version/0.22/generated/pandas.DataFrame.rolling.html)

Let's take a look at a quick sample.  


```python
close.rolling(window = 3)
```




    Rolling [window=3,center=False,axis=0]



This returns a Rolling object. Just like what you've seen before, it's an intermediate object similar to the GroupBy object which breaks the original data into groups. That means, we'll have to apply an operation to these groups. Let's try with sum function.


```python
close.rolling(window = 3).sum()
```




    2018-10-10     NaN
    2018-10-11     NaN
    2018-10-12     3.0
    2018-10-13     6.0
    2018-10-14     9.0
    2018-10-15    12.0
    2018-10-16    15.0
    2018-10-17    18.0
    2018-10-18    21.0
    2018-10-19    24.0
    2018-10-20    27.0
    Freq: D, dtype: float64



The window parameter defines the size of the moving window. This is the number of observations used for calculating the statistics which is the "sum" in our case.

For example, the output for 2018-10-12 is 3, which equals to the sum of the previous 3 data points, 0 + 1 + 2.
Another example is 2018-10-20 is 27, which equals to 8+ 9 + 10

Not just for summation, we can also apply other functions that we've learned in the previous lessons, such as max, min or even more.  

Let's have a look at another quick example


```python
close.rolling(window = 3).min()
```




    2018-10-10    NaN
    2018-10-11    NaN
    2018-10-12    0.0
    2018-10-13    1.0
    2018-10-14    2.0
    2018-10-15    3.0
    2018-10-16    4.0
    2018-10-17    5.0
    2018-10-18    6.0
    2018-10-19    7.0
    2018-10-20    8.0
    Freq: D, dtype: float64



Now, the output returns the minimum of the past three data points. 

By the way, have you noticed that we are getting NaN for close.rolling(window = 3).sum().  Since we are asking to calculate the mininum of the past 3 data points.  For 2018-10-10 and 2018-10-11, there are no enough data points in the past for our calculation, that's why we get NaN as outputs.  

There are many other parameters you can play with for this rolling function, such as min_period or so.  Please refer to [the python documentation](https://pandas.pydata.org/pandas-docs/version/0.22/generated/pandas.DataFrame.rolling.html) for more details


## Quiz: Calculate Simple Moving Average

Through out the program, you will learn to generate alpha factors.  However, signals are always noisy.  A common practise from the industry is to smooth the factors by using simple moving average.  In this quiz, we can create a simple function that you can specify the rolling window and calculate the simple moving average of a time series.  


```python
import quiz_tests

def calculate_simple_moving_average(rolling_window, close):
    """
    Compute the simple moving average.
    
    Parameters
    ----------
    rolling_window: int
        Rolling window length
    close : DataFrame
        Close prices for each ticker and date
    
    
    Returns
    -------
    simple_moving_average : DataFrame
        Simple moving average for each ticker and date
    """
    # TODO: Implement Function
    print(close.head(5))
    df=close.apply(lambda x : x.rolling(window = rolling_window).mean(), axis =0)
    print(df.head(5))
    return df


quiz_tests.test_calculate_simple_moving_average(calculate_simple_moving_average)
```

                       EAHB         WAE          BZSZ         SRPI         BEV
    2010-02-02  21.05081048 17.01384381   10.98450376  11.24809343 12.96171273
    2010-02-03  15.63570259 14.69054309   11.35302769 475.74195118 11.95964043
    2010-02-04 482.34539247 35.20258059 3516.54167823  66.40531433 13.50396048
    2010-02-05  10.91893302 17.90864387   24.80126542  12.48895419 10.52435923
    2010-02-06  10.67597197 12.74940144   11.80525758  21.53903949 19.99766037
                       EAHB         WAE          BZSZ         SRPI         BEV
    2010-02-02          nan         nan           nan          nan         nan
    2010-02-03          nan         nan           nan          nan         nan
    2010-02-04 173.01063518 22.30232250 1179.62640322 184.46511965 12.80843788
    2010-02-05 169.63334269 22.60058918 1184.23199044 184.87873990 11.99598671
    2010-02-06 167.98009915 21.95354197 1184.38273374  33.47776934 14.67532669
    Tests Passed


## Quiz Solution
If you're having trouble, you can check out the quiz solution [here](rolling_windows_solution.ipynb).


```python

```
