import numpy as np


class LinearRegression:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.x_mean = np.mean(x)
        self.y_mean = np.mean(y)
        self.gradient = self._get_gradient()
        self.constant = self._get_constant()

    def _get_gradient(self):
        """
        Calculates the gradient of a linear regression line.
        """
        numerator = sum(
            (self.x[i] - self.x_mean) * (self.y[i] - self.y_mean)
            for i in range(len(self.x))
        )
        denominator = sum((self.x[i] - self.x_mean) ** 2 for i in range(len(self.x)))
        return numerator / denominator

    def _get_constant(self):
        """
        Calculates the y-intercept of the linear regression line.
        """
        return self.y_mean - self.gradient * self.x_mean

    def _expected_y(self):
        """
        Returns the y values when the actual x values are
        entered into the linear regression equation.
            y = gradient * x + constant
        """
        return self.gradient * self.x + self.constant

    def _expected_x(self):
        """
        Returns the x values when the actual y values are
        entered into the linear regression equation.
            x = (y - constant) / gradient
        """
        return (self.y - self.constant) / self.gradient

    def r_squared(self):
        """
        Calculates R squared to see how well the regression
        line fits the data.
        Ranges from 0 - 1
        The closer to 1 the better the fit.
        """
        # Equation from: https://www.khanacademy.org/math/statistics-probability/describing-relationships-quantitative-data/more-on-regression/v/calculating-r-squared
        sum_squared_error = sum((self._expected_y() - self.y) ** 2)
        sum_squared_dist_from_mean = sum((self.y - self.y_mean) ** 2)
        return 1 - (sum_squared_error / sum_squared_dist_from_mean)


class ExponentialRegression:
    """
    Find linear regression equation and convert back to
    exponential equation.
    y = a * b**t => exponential standard equation.
    a = y-intercept
    b = 1 + growthrate
    t = time or period

    ln(y) = ln(a) + ln(b) * t
    Y = ln(y) => y = e**Y
    c = ln(a) => a = e**c
    m = ln(b) => b = e**m = 1 + growthrate
    Therefore: Y = m * t + c => linear standard equation
    """

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.log_y = np.log(y)

        lr = LinearRegression(self.x, self.log_y)
        self.m = lr.gradient
        self.c = lr.constant
        self.r_squared = lr.r_squared()

    def growth_rate(self):
        """
        Calculates growthrate of the exponential regression
        equation if: y = a * (1 + growthrate)**t
        """
        return np.e ** self.m - 1

    def expected_y(self, x):
        """
        Calculates y based on a given x value
        """
        return np.e * (self.m * x + self.c)
