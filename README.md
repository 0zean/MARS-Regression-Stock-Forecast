# MARS Regression Stock Forecast
 Forecasting the next day stock price for a given asset utilizing the multivariate adaptive regression splines algorithm and web scraping to get the VIX percent change.


 The Stock Data is first fetched using the Yahoo Finance API through the pandas-datareader library. 

 Using the BeautifulSoup webscraping library, the VIX percent change is retrieved from Yahoo Finance's page to offer further diretional confidence (VIX % change has an inverse relationship with The S&P 500 % change which can be seen by their [negative correlation](https://www.macroption.com/vix-spx-correlation/)). 

 The Stock data is then pre-processed to remove any unnecessary columns and split into testing and training sets. The open price is used to calculate a few technical indicators to be used in the regression analysis (Hilbert Transform Dominant Cycle period and Rolling Standard Deviation). Any missing values are imputed using k-nearest neighbors from the fancyimpute library.

 The resulting data is transformed using a log(1+features) transform to mitigate any exponential trends in the data and to account for the rightward skew in the data's distribution.

 The Feature set "X" consists of the 2 technical indicators along with the open price of the stock, and the target "Y" is the high, low, and close price, each fitted to their own regression model.

 A MARS regression is then fitted to each of the high, low and close prices separately. Adaboost regression is also used to better predict hard cases in each target series. The respective models are then used to predict the out of sample testing set, and directional accuracy is evaluated on each, being displayed in the final console output.

 To run the program, navigate to its directory and open a command prompt. Then enter "python mars_forecast.py". As input, enter the stock ticker and opening price for the day.


Here is an example of the visual output after running the script:

<img src="https://github.com/0zean/MARS-Regression-Stock-Forecast/blob/main/Visual.png">