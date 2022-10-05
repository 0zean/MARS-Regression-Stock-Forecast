# MARS Regression Stock Forecast
 Forecasting the next day stock price for a given asset utilizing the multivariate adaptive regression splines algorithm and web scraping to get the VIX percent change.


 The Stock Data is first fetched using the Yahoo Finance api through the pandas-datareader library. 

 Using the BeautifulSoup webscraping library, the VIX percent change is retrieved from Yahoo Finance's page to offer additional investment criteria (if the VIX % change is positive this often signifies a down turn in the overall market, if it is negative the opposite is true). 

 The Stock data is then pre-processed to remove any unecessary columns and split into testing and training sets. The open price is used to calculate a few technical indicators to be used in the regression analysis (Hilbert Transform Dominant Cycle period and Rolling Standard Deviation). Any missing values are imputed using k-nearest neighbors from the fancyimpute library.

 The reulting data is transformed using a log(1+features) transform to mitigrate any exponential trends in the data and to account for the rightward skew in the data's distribution.

 The Feature set "X" consists of the 2 technical indicators along with the open price of the stock, and the target "Y" is the high, low, and close price, each fitted to their own regression model.

 A MARS regression is then fitted to each of the high, low and close prices separately. 



Here is an example of the visual output after running the script:

<img src="https://github.com/0zean/MARS-Regression-Stock-Forecast/blob/main/Visual.png">