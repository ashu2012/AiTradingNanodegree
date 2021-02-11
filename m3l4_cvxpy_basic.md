
# Portfolio Optimization using cvxpy

## Install cvxpy and other libraries


```python
import sys
!{sys.executable} -m pip install -r requirements.txt
```

    Requirement already satisfied: colour==0.1.5 in /opt/conda/lib/python3.6/site-packages (from -r requirements.txt (line 1)) (0.1.5)
    Collecting cvxpy==1.0.3 (from -r requirements.txt (line 2))
    [?25l  Downloading https://files.pythonhosted.org/packages/a1/59/2613468ffbbe3a818934d06b81b9f4877fe054afbf4f99d2f43f398a0b34/cvxpy-1.0.3.tar.gz (880kB)
    [K    100% |████████████████████████████████| 880kB 4.6MB/s eta 0:00:01    98% |███████████████████████████████▋| 870kB 12.5MB/s eta 0:00:01
    [?25hRequirement already satisfied: cycler==0.10.0 in /opt/conda/lib/python3.6/site-packages/cycler-0.10.0-py3.6.egg (from -r requirements.txt (line 3)) (0.10.0)
    Collecting numpy==1.14.5 (from -r requirements.txt (line 4))
    [?25l  Downloading https://files.pythonhosted.org/packages/68/1e/116ad560de97694e2d0c1843a7a0075cc9f49e922454d32f49a80eb6f1f2/numpy-1.14.5-cp36-cp36m-manylinux1_x86_64.whl (12.2MB)
    [K    100% |████████████████████████████████| 12.2MB 1.3MB/s ta 0:00:011  4% |█▎                              | 491kB 13.9MB/s eta 0:00:01    8% |██▊                             | 1.1MB 13.4MB/s eta 0:00:01    13% |████▏                           | 1.6MB 13.0MB/s eta 0:00:01    21% |███████                         | 2.7MB 13.2MB/s eta 0:00:01    30% |█████████▋                      | 3.7MB 11.4MB/s eta 0:00:01    34% |██████████▉                     | 4.1MB 11.9MB/s eta 0:00:01    37% |████████████                    | 4.6MB 8.1MB/s eta 0:00:01    41% |█████████████▍                  | 5.1MB 10.0MB/s eta 0:00:01    49% |████████████████                | 6.1MB 14.8MB/s eta 0:00:01    54% |█████████████████▍              | 6.6MB 9.7MB/s eta 0:00:01    62% |███████████████████▉            | 7.6MB 8.2MB/s eta 0:00:01    66% |█████████████████████▎          | 8.1MB 23.0MB/s eta 0:00:01    70% |██████████████████████▊         | 8.6MB 11.4MB/s eta 0:00:01    87% |████████████████████████████    | 10.6MB 11.5MB/s eta 0:00:01    90% |█████████████████████████████   | 11.1MB 9.0MB/s eta 0:00:01    95% |██████████████████████████████▌ | 11.6MB 13.5MB/s eta 0:00:01    99% |███████████████████████████████▊| 12.1MB 10.9MB/s eta 0:00:01
    [?25hCollecting pandas==0.21.1 (from -r requirements.txt (line 5))
    [?25l  Downloading https://files.pythonhosted.org/packages/3a/e1/6c514df670b887c77838ab856f57783c07e8760f2e3d5939203a39735e0e/pandas-0.21.1-cp36-cp36m-manylinux1_x86_64.whl (26.2MB)
    [K    100% |████████████████████████████████| 26.2MB 638kB/s ta 0:00:011  0% |▎                               | 204kB 11.7MB/s eta 0:00:03    2% |█                               | 757kB 9.6MB/s eta 0:00:03    4% |█▋                              | 1.3MB 11.3MB/s eta 0:00:03    14% |████▊                           | 3.9MB 12.8MB/s eta 0:00:02    16% |█████▍                          | 4.4MB 10.2MB/s eta 0:00:03    18% |██████                          | 4.9MB 12.3MB/s eta 0:00:02    20% |██████▋                         | 5.4MB 12.9MB/s eta 0:00:02    24% |███████▉                        | 6.4MB 9.3MB/s eta 0:00:03    26% |████████▌                       | 6.9MB 11.1MB/s eta 0:00:02    28% |█████████                       | 7.4MB 7.6MB/s eta 0:00:03    30% |█████████▋                      | 7.9MB 9.1MB/s eta 0:00:03    33% |██████████▉                     | 8.9MB 9.2MB/s eta 0:00:02    35% |███████████▌                    | 9.4MB 8.1MB/s eta 0:00:03    37% |████████████                    | 9.9MB 8.3MB/s eta 0:00:02    41% |█████████████▏                  | 10.8MB 8.9MB/s eta 0:00:02    42% |█████████████▊                  | 11.2MB 9.3MB/s eta 0:00:02    46% |███████████████                 | 12.2MB 16.2MB/s eta 0:00:01    50% |████████████████▏               | 13.2MB 12.7MB/s eta 0:00:02    52% |████████████████▊               | 13.7MB 9.5MB/s eta 0:00:02    56% |██████████████████              | 14.8MB 12.1MB/s eta 0:00:01    59% |███████████████████▏            | 15.7MB 9.0MB/s eta 0:00:02    67% |█████████████████████▌          | 17.6MB 11.2MB/s eta 0:00:01    73% |███████████████████████▋        | 19.3MB 9.7MB/s eta 0:00:01    75% |████████████████████████▏       | 19.8MB 19.7MB/s eta 0:00:01    82% |██████████████████████████▎     | 21.6MB 21.1MB/s eta 0:00:01    84% |███████████████████████████     | 22.1MB 10.8MB/s eta 0:00:01    85% |███████████████████████████▍    | 22.5MB 8.4MB/s eta 0:00:01    98% |███████████████████████████████▍| 25.7MB 10.4MB/s eta 0:00:01    99% |████████████████████████████████| 26.2MB 21.5MB/s eta 0:00:01
    [?25hCollecting plotly==2.2.3 (from -r requirements.txt (line 6))
    [?25l  Downloading https://files.pythonhosted.org/packages/99/a6/8214b6564bf4ace9bec8a26e7f89832792be582c042c47c912d3201328a0/plotly-2.2.3.tar.gz (1.1MB)
    [K    100% |████████████████████████████████| 1.1MB 7.5MB/s ta 0:00:011 4% |█▌                              | 51kB 6.9MB/s eta 0:00:01    45% |██████████████▌                 | 491kB 11.0MB/s eta 0:00:01    87% |████████████████████████████▏   | 952kB 10.1MB/s eta 0:00:01
    [?25hRequirement already satisfied: pyparsing==2.2.0 in /opt/conda/lib/python3.6/site-packages (from -r requirements.txt (line 7)) (2.2.0)
    Requirement already satisfied: python-dateutil==2.6.1 in /opt/conda/lib/python3.6/site-packages (from -r requirements.txt (line 8)) (2.6.1)
    Requirement already satisfied: pytz==2017.3 in /opt/conda/lib/python3.6/site-packages (from -r requirements.txt (line 9)) (2017.3)
    Requirement already satisfied: requests==2.18.4 in /opt/conda/lib/python3.6/site-packages (from -r requirements.txt (line 10)) (2.18.4)
    Collecting scipy==1.0.0 (from -r requirements.txt (line 11))
    [?25l  Downloading https://files.pythonhosted.org/packages/d8/5e/caa01ba7be11600b6a9d39265440d7b3be3d69206da887c42bef049521f2/scipy-1.0.0-cp36-cp36m-manylinux1_x86_64.whl (50.0MB)
    [K    100% |████████████████████████████████| 50.0MB 365kB/s ta 0:00:011  0% |▎                               | 481kB 9.8MB/s eta 0:00:06    1% |▋                               | 942kB 11.7MB/s eta 0:00:05    3% |█▏                              | 1.9MB 10.1MB/s eta 0:00:05    8% |██▋                             | 4.2MB 11.2MB/s eta 0:00:05    12% |███▉                            | 6.1MB 10.3MB/s eta 0:00:05    14% |████▋                           | 7.2MB 8.9MB/s eta 0:00:05    15% |████▉                           | 7.6MB 9.5MB/s eta 0:00:05    17% |█████▊                          | 8.9MB 10.8MB/s eta 0:00:04    18% |██████                          | 9.4MB 8.8MB/s eta 0:00:05    19% |██████▎                         | 9.8MB 8.5MB/s eta 0:00:05    20% |██████▌                         | 10.2MB 9.4MB/s eta 0:00:05    21% |██████▉                         | 10.7MB 9.3MB/s eta 0:00:05    23% |███████▍                        | 11.6MB 7.6MB/s eta 0:00:06    23% |███████▊                        | 12.0MB 10.1MB/s eta 0:00:04    24% |████████                        | 12.4MB 7.6MB/s eta 0:00:05    28% |█████████                       | 14.2MB 11.3MB/s eta 0:00:04    29% |█████████▎                      | 14.5MB 6.8MB/s eta 0:00:06    30% |█████████▋                      | 15.0MB 8.6MB/s eta 0:00:05    30% |█████████▉                      | 15.4MB 7.5MB/s eta 0:00:05    31% |██████████▎                     | 16.0MB 11.0MB/s eta 0:00:04    33% |██████████▊                     | 16.7MB 8.1MB/s eta 0:00:05    34% |███████████                     | 17.2MB 8.9MB/s eta 0:00:04    36% |███████████▋                    | 18.1MB 10.7MB/s eta 0:00:03    37% |███████████▉                    | 18.6MB 10.5MB/s eta 0:00:03    37% |████████████▏                   | 19.0MB 8.9MB/s eta 0:00:04    41% |█████████████▍                  | 20.9MB 7.7MB/s eta 0:00:04    42% |█████████████▋                  | 21.4MB 11.3MB/s eta 0:00:03    43% |██████████████                  | 21.7MB 10.0MB/s eta 0:00:03    44% |██████████████▏                 | 22.1MB 5.4MB/s eta 0:00:06    44% |██████████████▍                 | 22.5MB 8.5MB/s eta 0:00:04    45% |██████████████▋                 | 22.9MB 7.9MB/s eta 0:00:04    46% |███████████████                 | 23.3MB 9.8MB/s eta 0:00:03    47% |███████████████▏                | 23.7MB 11.1MB/s eta 0:00:03    49% |███████████████▊                | 24.6MB 11.8MB/s eta 0:00:03    50% |████████████████                | 25.0MB 7.3MB/s eta 0:00:04    50% |████████████████▎               | 25.5MB 11.5MB/s eta 0:00:03    54% |█████████████████▍              | 27.1MB 7.4MB/s eta 0:00:04    55% |█████████████████▊              | 27.6MB 7.5MB/s eta 0:00:03    56% |██████████████████              | 28.0MB 7.2MB/s eta 0:00:04    57% |██████████████████▌             | 28.8MB 9.3MB/s eta 0:00:03    60% |███████████████████▎            | 30.1MB 8.6MB/s eta 0:00:03    61% |███████████████████▋            | 30.7MB 10.9MB/s eta 0:00:02    63% |████████████████████▌           | 32.0MB 11.6MB/s eta 0:00:02    64% |████████████████████▊           | 32.5MB 6.8MB/s eta 0:00:03    65% |█████████████████████           | 32.8MB 9.8MB/s eta 0:00:02    66% |█████████████████████▎          | 33.2MB 10.8MB/s eta 0:00:02    67% |█████████████████████▌          | 33.7MB 8.8MB/s eta 0:00:02    71% |██████████████████████▉         | 35.6MB 10.0MB/s eta 0:00:02    72% |███████████████████████         | 36.1MB 10.7MB/s eta 0:00:02    73% |███████████████████████▊        | 37.0MB 7.8MB/s eta 0:00:02    74% |████████████████████████        | 37.5MB 11.1MB/s eta 0:00:02    76% |████████████████████████▌       | 38.4MB 7.7MB/s eta 0:00:02    79% |█████████████████████████▍      | 39.7MB 8.2MB/s eta 0:00:02    80% |██████████████████████████      | 40.5MB 9.2MB/s eta 0:00:02    84% |███████████████████████████     | 42.2MB 11.1MB/s eta 0:00:01    85% |███████████████████████████▏    | 42.5MB 6.4MB/s eta 0:00:02    85% |███████████████████████████▌    | 43.0MB 11.1MB/s eta 0:00:01    86% |███████████████████████████▊    | 43.4MB 13.0MB/s eta 0:00:01    87% |████████████████████████████    | 43.8MB 8.5MB/s eta 0:00:01    88% |████████████████████████████▎   | 44.2MB 10.6MB/s eta 0:00:01    89% |████████████████████████████▋   | 44.6MB 8.5MB/s eta 0:00:01    90% |████████████████████████████▉   | 45.0MB 7.1MB/s eta 0:00:01    90% |█████████████████████████████   | 45.5MB 7.8MB/s eta 0:00:01    91% |█████████████████████████████▍  | 45.9MB 7.5MB/s eta 0:00:01    92% |█████████████████████████████▋  | 46.3MB 7.1MB/s eta 0:00:01    93% |██████████████████████████████  | 46.8MB 9.5MB/s eta 0:00:01    94% |██████████████████████████████▏ | 47.2MB 6.2MB/s eta 0:00:01    95% |██████████████████████████████▊ | 48.0MB 8.5MB/s eta 0:00:01    96% |███████████████████████████████ | 48.5MB 9.2MB/s eta 0:00:01    98% |███████████████████████████████▌| 49.3MB 6.0MB/s eta 0:00:01    99% |███████████████████████████████▉| 49.7MB 10.1MB/s eta 0:00:01
    [?25hRequirement already satisfied: scikit-learn==0.19.1 in /opt/conda/lib/python3.6/site-packages (from -r requirements.txt (line 12)) (0.19.1)
    Requirement already satisfied: six==1.11.0 in /opt/conda/lib/python3.6/site-packages (from -r requirements.txt (line 13)) (1.11.0)
    Collecting tqdm==4.19.5 (from -r requirements.txt (line 14))
    [?25l  Downloading https://files.pythonhosted.org/packages/71/3c/341b4fa23cb3abc335207dba057c790f3bb329f6757e1fcd5d347bcf8308/tqdm-4.19.5-py2.py3-none-any.whl (51kB)
    [K    100% |████████████████████████████████| 61kB 5.0MB/s ta 0:00:01
    [?25hCollecting osqp (from cvxpy==1.0.3->-r requirements.txt (line 2))
    [?25l  Downloading https://files.pythonhosted.org/packages/76/82/b0693a167e4b9b5e94f4988f6df3d7866e9e41a316a58f1053dd21370f1a/osqp-0.6.2.post0-cp36-cp36m-manylinux1_x86_64.whl (211kB)
    [K    100% |████████████████████████████████| 215kB 6.7MB/s ta 0:00:011  24% |███████▊                        | 51kB 5.2MB/s eta 0:00:01
    [?25hCollecting ecos>=2 (from cvxpy==1.0.3->-r requirements.txt (line 2))
    [?25l  Downloading https://files.pythonhosted.org/packages/55/ed/d131ff51f3a8f73420eb1191345eb49f269f23cadef515172e356018cde3/ecos-2.0.7.post1-cp36-cp36m-manylinux1_x86_64.whl (147kB)
    [K    100% |████████████████████████████████| 153kB 4.5MB/s ta 0:00:01   34% |███████████▏                    | 51kB 5.2MB/s eta 0:00:01
    [?25hCollecting scs>=1.1.3 (from cvxpy==1.0.3->-r requirements.txt (line 2))
    [?25l  Downloading https://files.pythonhosted.org/packages/1a/72/33be87cce255d4e9dbbfef547e9fd6ec7ee94d0d0910bb2b13badea3fbbe/scs-2.1.2.tar.gz (3.5MB)
    [K    100% |████████████████████████████████| 3.6MB 4.9MB/s ta 0:00:011 0% |▏                               | 20kB 6.7MB/s eta 0:00:01    13% |████▍                           | 481kB 9.5MB/s eta 0:00:01    52% |█████████████████               | 1.9MB 11.3MB/s eta 0:00:01    78% |█████████████████████████▏      | 2.8MB 10.5MB/s eta 0:00:01    92% |█████████████████████████████▌  | 3.3MB 9.0MB/s eta 0:00:01
    [?25hCollecting multiprocess (from cvxpy==1.0.3->-r requirements.txt (line 2))
    [?25l  Downloading https://files.pythonhosted.org/packages/8f/dc/426a82723c460cfab653ebb717590103d6e38cebc9d1f599b0898915ac1d/multiprocess-0.70.11.1-py36-none-any.whl (101kB)
    [K    100% |████████████████████████████████| 102kB 3.3MB/s a 0:00:01
    [?25hRequirement already satisfied: fastcache in /opt/conda/lib/python3.6/site-packages (from cvxpy==1.0.3->-r requirements.txt (line 2)) (1.0.2)
    Requirement already satisfied: toolz in /opt/conda/lib/python3.6/site-packages (from cvxpy==1.0.3->-r requirements.txt (line 2)) (0.8.2)
    Requirement already satisfied: decorator>=4.0.6 in /opt/conda/lib/python3.6/site-packages (from plotly==2.2.3->-r requirements.txt (line 6)) (4.0.11)
    Requirement already satisfied: nbformat>=4.2 in /opt/conda/lib/python3.6/site-packages (from plotly==2.2.3->-r requirements.txt (line 6)) (4.4.0)
    Requirement already satisfied: chardet<3.1.0,>=3.0.2 in /opt/conda/lib/python3.6/site-packages (from requests==2.18.4->-r requirements.txt (line 10)) (3.0.4)
    Requirement already satisfied: idna<2.7,>=2.5 in /opt/conda/lib/python3.6/site-packages (from requests==2.18.4->-r requirements.txt (line 10)) (2.6)
    Requirement already satisfied: urllib3<1.23,>=1.21.1 in /opt/conda/lib/python3.6/site-packages (from requests==2.18.4->-r requirements.txt (line 10)) (1.22)
    Requirement already satisfied: certifi>=2017.4.17 in /opt/conda/lib/python3.6/site-packages (from requests==2.18.4->-r requirements.txt (line 10)) (2019.11.28)
    Collecting qdldl (from osqp->cvxpy==1.0.3->-r requirements.txt (line 2))
    [?25l  Downloading https://files.pythonhosted.org/packages/ec/a3/db0e7c9fec5387dc33cbd2819329c141ba76497148aa9fab4bd1a7c2a279/qdldl-0.1.5.post0.tar.gz (69kB)
    [K    100% |████████████████████████████████| 71kB 6.8MB/s ta 0:00:011
    [?25hCollecting dill>=0.3.3 (from multiprocess->cvxpy==1.0.3->-r requirements.txt (line 2))
    [?25l  Downloading https://files.pythonhosted.org/packages/52/d6/79f40d230895fa1ce3b6af0d22e0ac79c65175dc069c194b79cc8e05a033/dill-0.3.3-py2.py3-none-any.whl (81kB)
    [K    100% |████████████████████████████████| 81kB 5.3MB/s ta 0:00:011
    [?25hRequirement already satisfied: jupyter-core in /opt/conda/lib/python3.6/site-packages (from nbformat>=4.2->plotly==2.2.3->-r requirements.txt (line 6)) (4.4.0)
    Requirement already satisfied: ipython-genutils in /opt/conda/lib/python3.6/site-packages (from nbformat>=4.2->plotly==2.2.3->-r requirements.txt (line 6)) (0.2.0)
    Requirement already satisfied: traitlets>=4.1 in /opt/conda/lib/python3.6/site-packages (from nbformat>=4.2->plotly==2.2.3->-r requirements.txt (line 6)) (4.3.2)
    Requirement already satisfied: jsonschema!=2.5.0,>=2.4 in /opt/conda/lib/python3.6/site-packages (from nbformat>=4.2->plotly==2.2.3->-r requirements.txt (line 6)) (2.6.0)
    Building wheels for collected packages: cvxpy, plotly, scs, qdldl
      Running setup.py bdist_wheel for cvxpy ... [?25ldone
    [?25h  Stored in directory: /root/.cache/pip/wheels/2b/60/0b/0c2596528665e21d698d6f84a3406c52044c7b4ca6ac737cf3
      Running setup.py bdist_wheel for plotly ... [?25ldone
    [?25h  Stored in directory: /root/.cache/pip/wheels/98/54/81/dd92d5b0858fac680cd7bdb8800eb26c001dd9f5dc8b1bc0ba
      Running setup.py bdist_wheel for scs ... [?25ldone
    [?25h  Stored in directory: /root/.cache/pip/wheels/df/d0/79/37ea880586da03c620ca9ecd5e42adbd86bc6ea84363965c5f
      Running setup.py bdist_wheel for qdldl ... [?25ldone
    [?25h  Stored in directory: /root/.cache/pip/wheels/a9/77/d6/726fc4a2ae1513b4663b81721f5d75e9b4fe9d74ca7a8a5417
    Successfully built cvxpy plotly scs qdldl
    [31mtensorflow 1.3.0 requires tensorflow-tensorboard<0.2.0,>=0.1.0, which is not installed.[0m
    [31mmoviepy 0.2.3.2 has requirement tqdm==4.11.2, but you'll have tqdm 4.19.5 which is incompatible.[0m
    Installing collected packages: numpy, scipy, qdldl, osqp, ecos, scs, dill, multiprocess, cvxpy, pandas, plotly, tqdm
      Found existing installation: numpy 1.12.1
        Uninstalling numpy-1.12.1:
          Successfully uninstalled numpy-1.12.1
      Found existing installation: scipy 1.2.1
        Uninstalling scipy-1.2.1:
          Successfully uninstalled scipy-1.2.1
      Found existing installation: dill 0.2.7.1
        Uninstalling dill-0.2.7.1:
          Successfully uninstalled dill-0.2.7.1
      Found existing installation: pandas 0.23.3
        Uninstalling pandas-0.23.3:
          Successfully uninstalled pandas-0.23.3
      Found existing installation: plotly 2.0.15
        Uninstalling plotly-2.0.15:
          Successfully uninstalled plotly-2.0.15
      Found existing installation: tqdm 4.11.2
        Uninstalling tqdm-4.11.2:
          Successfully uninstalled tqdm-4.11.2
    Successfully installed cvxpy-1.0.3 dill-0.3.3 ecos-2.0.7.post1 multiprocess-0.70.11.1 numpy-1.14.5 osqp-0.6.2.post0 pandas-0.21.1 plotly-2.2.3 qdldl-0.1.5.post0 scipy-1.0.0 scs-2.1.2 tqdm-4.19.5


## Imports


```python
import cvxpy as cvx
import numpy as np
import quiz_tests
```

## Optimization with cvxpy

http://www.cvxpy.org/

Practice using cvxpy to solve a simple optimization problem. Find the optimal weights on a two-asset portfolio given the variance of Stock A, the variance of Stock B, and the correlation between Stocks A and B. Create a function that takes in these values as arguments and returns the vector of optimal weights, i.e., 

$\mathbf{x} = \begin{bmatrix}
x_A & x_B
\end{bmatrix}
$


Remember that the constraint in this problem is: $x_A + x_B = 1$



## Hints

### standard deviation
standard deviation $\sigma_A = \sqrt(\sigma^2_A)$, where $\sigma^2_A$ is variance of $x_A$
look at `np.sqrt()`

### covariance
correlation between the stocks is $\rho_{A,B}$

covariance between the stocks is $\sigma_{A,B} = \sigma_A \times \sigma_B \times \rho_{A,B}$

### x vector
create a vector of 2 x variables $\mathbf{x} = \begin{bmatrix}
x_A & x_B
\end{bmatrix}
$
we can use `cvx.Variable(2)`

### covariance matrix
The covariance matrix $P = 
\begin{bmatrix}
\sigma^2_A & \sigma_{A,B} \\ 
\sigma_{A,B} & \sigma^2_B 
\end{bmatrix}$

We can create a 2 x 2 matrix using a 2-dimensional numpy array
`np.array([["Cindy", "Liz"],["Eddy", "Brok"]])`

### quadratic form
We can write the portfolio variance $\sigma^2_p = \mathbf{x^T} \mathbf{P} \mathbf{x}$

Recall that the $\mathbf{x^T} \mathbf{P} \mathbf{x}$ is called the quadratic form.
We can use the cvxpy function `quad_form(x,P)` to get the quadratic form.

### objective function
Next, we want to define the objective function.  In this case, we want to minimize something.  What do we want to minimize in this case?  We want to minimize the portfolio variance, which is defined by our quadratic form $\mathbf{x^T} \mathbf{P} \mathbf{x}$

We can find the objective function using cvxpy `objective = cvx.Minimize()`.  Can you guess what to pass into this function?


### constraints
We can also define our constraints in a list.  For example, if you wanted the $\sum_{1}^{n}x = 1$, you could save a variable as `[sum(x)==1]`, where x was created using `cvx.Variable()`.

### optimization
So now that we have our objective function and constraints, we can solve for the values of $\mathbf{x}$.
cvxpy has the constructor `Problem(objective, constraints)`, which returns a `Problem` object.

The `Problem` object has a function solve(), which returns the minimum of the solution.  In this case, this is the minimum variance of the portfolio.

It also updates the vector $\mathbf{x}$.

We can check out the values of $x_A$ and $x_B$ that gave the minimum portfolio variance by using `x.value`


```python
import cvxpy as cvx
import numpy as np

def optimize_twoasset_portfolio(varA, varB, rAB):
    """Create a function that takes in the variance of Stock A, the variance of
    Stock B, and the correlation between Stocks A and B as arguments and returns 
    the vector of optimal weights
    
    Parameters
    ----------
    varA : float
        The variance of Stock A.
        
    varB : float
        The variance of Stock B.    
        
    rAB : float
        The correlation between Stocks A and B.
        
    Returns
    -------
    x : np.ndarray
        A 2-element numpy ndarray containing the weights on Stocks A and B,
        [x_A, x_B], that minimize the portfolio variance.
    
    """
    # TODO: Use cvxpy to determine the weights on the assets in a 2-asset
    # portfolio that minimize portfolio variance.
    
    #cov = 
    
    # x = 
    
    # P = 
    
    #objective = 
    
    # constraints = 
    
    # problem = 
    
    #min_value = 
    # xA,xB = 
    
    # return xA and xB
    cov = np.sqrt(varA)*np.sqrt(varB)*rAB
    x = cvx.Variable(2)
    P = np.array([[varA, cov],[cov, varB]])
    objective = cvx.Minimize(cvx.quad_form(x,P))
    constraints = [sum(x)==1]
    problem = cvx.Problem(objective, constraints)
    min_value = problem.solve()
    xA,xB = x.value
    
    return xA, xB
    

quiz_tests.test_optimize_twoasset_portfolio(optimize_twoasset_portfolio)
```

    Tests Passed



```python
"""Test run optimize_twoasset_portfolio()."""
xA,xB = optimize_twoasset_portfolio(0.1, 0.05, 0.25)
print("Weight on Stock A: {:.6f}".format(xA))
print("Weight on Stock B: {:.6f}".format(xB))
```

    Weight on Stock A: 0.281935
    Weight on Stock B: 0.718065


If you're feeling stuck, you can check out the solution [here](m3l4_cvxpy_basic_solution.ipynb)


```python

```
