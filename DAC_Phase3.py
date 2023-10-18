import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


def load_and_preprocess_data(filename):
    data = pd.read_csv(filename)
    data['dateRep'] = pd.to_datetime(data['dateRep'])
    return data

def select_features_and_target(data):
    X = data[['day', 'month', 'year']].values
    y = data['deaths'].values
    return X, y


def build_linear_regression_model(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model

def predict_max_death_date(data, model):
    predicted_deaths = model.predict(data[['day', 'month', 'year']].values)
    max_death_index = predicted_deaths.argmax()
    max_death_date = data.iloc[max_death_index]['dateRep']
    max_death_deaths = predicted_deaths[max_death_index]
    return max_death_date, max_death_deaths

def visualize_data(data, predicted_deaths):
    plt.figure(figsize=(10, 6))
    plt.scatter(data['dateRep'], data['deaths'], label='Actual Deaths')
    plt.plot(data['dateRep'], predicted_deaths, 'r', label='Predicted Deaths')
    plt.xlabel('Date')
    plt.ylabel('Deaths')
    plt.title('Actual vs. Predicted Deaths')
    plt.legend()
    plt.show()


def main():
    filename = 'Covid_19_cases4.csv'  
    data = load_and_preprocess_data(filename)
    X, y = select_features_and_target(data)
    model = build_linear_regression_model(X, y)
    max_death_date, max_death_deaths = predict_max_death_date(data, model)

    print(f"Date with the maximum predicted deaths: {max_death_date}")
    print(f"Predicted deaths on {new_date}: {predicted_deaths[0]}")

    visualize_data(data, model.predict(X))

if __name__ == "__main__":
    main()
