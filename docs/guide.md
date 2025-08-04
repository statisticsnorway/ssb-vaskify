# Guide to vaskify

The package `vaskify` has been created to help with data control and editing processes for establishment surveys. It contains a limited number of functions but can be expanded as further needs are defined.

## Installation
The package is on PyPI can be installed in a poetry environment by running the following in a terminal:
```bash
poetry add ssb-vaskify
```

## Import error detection functions (`Detect`)
The main module includes functions for detecting potential errors (outliers). The main class used is called `Detect` and can be imported with:
```python
from vaskify import Detect
```

## Establish the class instance
Use your own data, or try our `create_test_data`function to create som random data to test the functions. Data can be created like this:

```python
testdata = create_test_data(10, n_periods=2, freq="yearly", seed=4)
```

We establish an object to allow use to apply different detection methods using the class `Detect`. The input is the data to check and the identification variable for the units in the data. The data should be in [long form](https://towardsdatascience.com/long-and-wide-formats-in-data-explained-e48d7c9a06cb) with time periods below one another.

```python
det = Detect(testdata, id_nr="id_company")
```

## Check for thousand errors
Sometimes, particularly in establishment surveys, thousand errors occur. This may occur when a company reports a value in actual dollars when asked for the value in thousands (or millions) of dollars. These errors can be check for using reporting from a previous time period. For example here we check for errors in the 'turnover' variable, based on the previous time period, using the varaiable 'time_period'.

```python
det.thousand_error(y_var="turnover", time_var="time_period")
```

## Check for accumulation errors
In panel establishment surveys, companies will sometimes accidentally report an accumulative amount for the year rather than the specified period. the `accumulation_error` function can be used to detect these cases.

```python
det.accumulation_error(y_var="turnover", time_var="time_period")
```

## Check for outliers using the HB-method
Hidiroglou-Berthelot (HB) method is a popular tool for detecting outliers in data in establishment surveys. It is a data driven approach to determine the parameters for edits. [Winkler et. al. ](http://www.asasrms.org/Proceedings/y2023/files/HB_JSM_2023.pdf) provide a nice summary evaluating the method.

```python
det.hb(y_var="turnover", time_var="time_period")
```
