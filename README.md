# RAPM

Basic NBA RAPM (Regularized Adjusted Plus Minus) code to work in conjunction with this https://x.com/JerryEngelmann/status/1797822392968065372 dataset.
Original paper on RAPM (Joe Sill, 2010) available here https://supermariogiacomazzo.github.io/STOR538_WEBSITE/Articles/Basketball/Basketball_Sill.pdf
First public description of APM (Dan Rosenbaum, 2004) here https://www.82games.com/comm30.htm

Performs a penalized (here, Ridge) regression with players (-offense, -defense) as explanatory variables, and points scored as the dependent variable
