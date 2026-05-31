# Formula 1 Qualifying vs Race Results

## Project Objective

This project analyzes the relationship between Formula 1 qualifying position and actual race finishing position.

The main goal is to determine whether a driver's qualifying result can help explain or predict their final race result using data analysis and simple linear regression.

## Main Project Question

**Does qualifying position have a meaningful relationship with race finishing position in Formula 1?**

## Supporting Questions

* Do drivers who qualify near the front usually finish near the front?
* How strong is the relationship between qualifying position and final race finish?
* Can qualifying position be used as a predictor of race finish position?
* What does a simple linear regression model reveal about this relationship?
* What are the limitations of using qualifying position alone to predict race results?

## Why This Project Matters

Formula 1 races are influenced by many factors, including driver skill, car performance, team strategy, pit stops, penalties, weather, safety cars, and mechanical reliability.

Even with these factors, qualifying position is often viewed as important because drivers who start closer to the front usually have a better opportunity to finish well.

This project uses real Formula 1 data to test that idea by analyzing the correlation between qualifying position and race finishing position, then building a simple linear regression model to evaluate how well qualifying position can predict race results.

## Key Findings

The analysis found a moderate positive relationship between qualifying position and race finishing position.

* Correlation between qualifying position and race finish position: 0.576
* Linear regression slope: 0.574
* Testing R² score: 0.352
* Mean Absolute Error: about 4 finishing positions
* RMSE: about 5.1 finishing positions

The positive correlation and regression slope suggest that drivers who qualify closer to the front generally tend to finish closer to the front. However, the relationship is not perfect.

The testing R² value shows that qualifying position explains about 35.2% of the variation in race finishing position. This means qualifying position is an important factor, but it does not fully determine the race result.

## Conclusion

This project shows that qualifying position can be used as a reasonable baseline predictor of Formula 1 race finishing position. Drivers who qualify better generally have a higher chance of finishing better, but race results are still influenced by many other factors.

The model’s average prediction error was about 4 finishing positions, which shows that qualifying position alone is not enough to accurately predict every race result. Factors such as pit strategy, penalties, crashes, safety cars, weather, car reliability, and race pace can all affect the final outcome.

Overall, the project supports the idea that qualifying performance matters in Formula 1, but a stronger prediction model would need additional variables beyond qualifying position.
