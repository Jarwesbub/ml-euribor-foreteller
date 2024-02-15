import joblib

# This is only a test for using Python's machine learning tools in realtime
# It reads a model created from csv and makes predictions from it
# All the results are only predictions based on given data and are far from accurate!

# Steps:
# 1. Run "euribor_model_creator"
# 2. Run this program and results will be printed in the console (in current test build)

class Predict:
    def __init__(self, day, month, year) :
        self.day = day
        self.month = month
        self.year = year
        self.predictions = self._predict_euribor_by_day()
        self.print_results()

    def _predict_euribor_by_day(self):
        model = joblib.load('euriborModel.joblib')
        euribors = [0.2, 1, 3, 6, 12]
        array = []
        for kk in euribors:
            array.append(model.predict([[kk, self.day, self.month, self.year]]))
        return array
    
    def print_results(self):
        date = str(self.day)+'.'+str(self.month)+'.'+str(self.year)
        print("Euribor predictions "+date)
        print("1 week: "+self.predictions[0])
        print("1 month: "+self.predictions[1])
        print("3 months: "+self.predictions[2])
        print("6 months: "+self.predictions[3])
        print("12 months: "+self.predictions[4])

# Given date for prediction from the user:
day = 10
month = 2
year = 2024

predict = Predict(day, month, year)