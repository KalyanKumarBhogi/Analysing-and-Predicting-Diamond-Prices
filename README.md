A diamond is one of the most expensive stones. The price of diamonds varies irrespective of the size because of the factors affecting the price of a diamond. <p>

# Diamond Price Analysis
To analyze the price of diamonds according to their attributes, we first need to have a dataset containing diamond prices based on their features. I found ideal data on Kaggle containing information about diamonds like: <p>
1.Carat <p>
2.Cut <p>
3.Colour <p>
4.Clarity <p>
5.Depth <p>
6.Table <p>
7.Price <p>
8.Size <p>

# Diamond Price Analysis using Python <p>

**Loading all the possible libaries**
import pandas as pd <p>
import numpy as np <p>
import plotly.express as px <p>
import plotly.graph_objects as go <p>
**Reading of Dataset** <p>
data = pd.read_csv("diamonds.csv") <p>
**Head of Dataset** <p>
data.head() <p>
**Tail of the Dataset** <p>
data.tail() <p>

**This dataset contains an Unnamed column. I will delete this column before moving further:** <p>
data = data.drop("Unnamed: 0",axis=1) <p>
**Descriptive Statistics** <p>
data.describe() <p>
**To find nulls in the dataset**<p>
data.isnull().sum()<p>

**Now let’s start analyzing diamond prices. I will first analyze the relationship between the carat and the price of the diamond to see how the number of carats affects the price of a diamond:** <p>

figure = px.scatter(data_frame = data, x="carat",
                    y="price", size="depth", 
                    color= "cut", trendline="ols")  <p>
figure.show() <p>

**We can see a linear relationship between the number of carats and the price of a diamond. It means higher carats result in higher prices.**

**Now I will add a new column to this dataset by calculating the size (length x width x depth) of the diamond:**

data["size"] = data["x"] * data["y"] * data["z"] <p>
print(data) <p>

**Now let’s have a look at the relationship between the size of a diamond and its price:** <p>
figure = px.scatter(data_frame = data, x="size",
                    y="price", size="size", 
                    color= "cut", trendline="ols") <p>
figure.show()  <p>

**The above figure concludes two features of diamonds:**
1. Premium-cut diamonds are relatively larger than other diamonds <p>
2. There’s a linear relationship between the size of all types of diamonds and their prices <p>

Now let’s have a look at the prices of all the types of diamonds based on their color: <p>

fig = px.box(data, x="cut", 
             y="price", 
             color="color")  <p>
fig.show() <p>

**Now let’s have a look at the prices of all the types of diamonds based on their clarity:** <p>

fig = px.box(data, 
             x="cut", 
             y="price", 
             color="clarity") <p>
fig.show() <p>

# Diamond Price Prediction <p>
Now, I will move to the task of predicting diamond prices by using all the necessary information from the diamond price analysis done above. <p>

Before moving forward, I will convert the values of the cut column as the cut type of diamonds is a valuable feature to predict the price of a diamond. To use this column, we need to convert its categorical values into numerical values. Below is how we can convert it into a numerical feature: <p>

data["cut"] = data["cut"].map({"Ideal": 1, 
                               "Premium": 2, 
                               "Good": 3,
                               "Very Good": 4,
                               "Fair": 5}) <p>

**Now, let’s split the data into training and test sets:**    <p>

# splitting data
from sklearn.model_selection import train_test_split  <p>
x = np.array(data[["carat", "cut", "size"]])  <p>
y = np.array(data[["price"]]) <p>

xtrain, xtest, ytrain, ytest = train_test_split(x, y, 
                                                test_size=0.10, 
                                                random_state=42) <p>

**Now I will train a machine learning model for the task of diamond price prediction:**     <p>

from sklearn.ensemble import RandomForestRegressor <p>
model = RandomForestRegressor() <p>
model.fit(xtrain, ytrain) <p>

**Now below is how we can use our machine learning model to predict the price of a diamond:** <p>

print("Diamond Price Prediction")  <p>
a = float(input("Carat Size: ")) <p>
b = int(input("Cut Type (Ideal: 1, Premium: 2, Good: 3, Very Good: 4, Fair: 5): ")) <p>
c = float(input("Size: ")) <p>
features = np.array([[a, b, c]]) <p>
print("Predicted Diamond's Price = ", model.predict(features)) <p>


# Summary
According to the diamond price analysis, we can say that the price and size of premium diamonds are higher than other types of diamonds.

