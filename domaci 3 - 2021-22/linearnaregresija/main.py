import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

#Problem statement and read data
pd.set_option("display.max_columns",15)
pd.set_option("display.width",None)
data=pd.read_csv("datasets/car_purchase.csv")
print(data)

#Data analysis
print(data.info())
#feature statistic
print(data.describe())
print(data.describe(include=[object]))

#Data cleansing


#Feature engineering
#useful features: age, salary, credit_car_debr, net_worth
#not-useful: customer_id
data_train=data.loc[:, ["gender", "age", "annual_salary", "credit_card_debt", "net_worth"]] #DataFrame
labels=data.loc[:,"max_purchase_amount"] #series
le=LabelEncoder()
data_train.gender=le.fit_transform(data_train.gender)
data_train.age=data_train.age/100
data_train.annual_salary=data_train.annual_salary/100000
data_train.credit_card_debt=data_train.credit_card_debt/100000
data_train.net_worth=data_train.net_worth/1000000


y=labels/100000
labels=labels/100000
plt.figure("gender")
plt.scatter(data_train.gender, y, s=23, color="red",label="people")
plt.xlabel('Gender 0-female, 1-male', fontsize=13)
plt.ylabel('Max purchase amount', fontsize=13)
plt.title('Gender per customer')
plt.legend()
plt.show()

plt.figure("age")
plt.scatter(data_train.age, y, s=23, color="blue",marker="x",label="people")
plt.xlabel('Age', fontsize=13)
plt.ylabel('Max purchase amount', fontsize=13)
plt.title('Age per customer')
plt.legend()
plt.show()

plt.figure("Annual salary")
plt.scatter(data_train.annual_salary, y, s=23, color="black",marker="o",label="people")
plt.xlabel('Annual salary', fontsize=13)
plt.ylabel('Max purchase amount', fontsize=13)
plt.title('Annual salary per customer')
plt.legend()
plt.show()

plt.figure("credit card debt")
plt.scatter(data_train.credit_card_debt, y, s=17, color="green",marker=".",label="people")
plt.xlabel('Credit card debt', fontsize=13)
plt.ylabel('Max purchase amount', fontsize=13)
plt.title('Credit card debt per customer per customer')
plt.legend()
plt.show()

plt.figure("net worth")
plt.scatter(data_train.net_worth, y, s=23, color="purple",marker="v",label="people")
plt.xlabel('Net worth', fontsize=13)
plt.ylabel('Max purchase amount', fontsize=13)
plt.title('Net worth per customer per customer')
plt.legend()
plt.show()

# Kreiranje i obucavanje sklearn.LinearRegression modela
xtrain, xtest, ytrain, ytest=train_test_split(data_train, labels, train_size=0.7, random_state=123, shuffle=False)
lr_model = LinearRegression()
lr_model.fit(xtrain, ytrain)
ypred=lr_model.predict(xtest)
print(f'LR MSE: {mean_squared_error(ytest, ypred):.2f}')
print(f'LR score: {lr_model.score(xtest, ytest):.2f}\n')

class LinearRegressionGradientDescent:
    def __init__(self):
        self.coeff = None
        self.features = None
        self.target = None
        self.mse_history = None

    def set_coefficients(self, *args):
    # Mapiramo koeficijente u niz oblika (n + 1) x 1
        self.coeff = np.array(args).reshape(-1, 1)

    def cost(self):
        predicted = self.features.dot(self.coeff)
        s = pow(predicted - self.target, 2).sum()
        return (0.5 / len(self.features)) * s

    # Argument mora biti DataFrame
    def predict(self, features):
        features = features.copy(deep=True)
        features.insert(0, 'c0', np.ones((len(features), 1)))
        features = features.to_numpy()
        return features.dot(self.coeff).reshape(-1, 1).flatten()

    def gradient_descent_step(self, learning_rate):
    # learning_rate - korak ucenja; dimenzije ((n + 1) x 1);
    # korak ucenja je razlicit za razlicite koeficijente
    # m - broj uzoraka
    # n - broj razlicitih atributa (osobina)
    # features – dimenzije (m x (n + 1));
    # n + 1 je zbog koeficijenta c0
    # self.coeff – dimenzije ((n + 1) x 1)
    # predicted – dimenzije (m x (n + 1)) x ((n + 1) x 1) = (m x 1)
        predicted = self.features.dot(self.coeff)
    # koeficijeni se azuriraju po formuli:
    # coeff(i) = coeff(i) - learning_rate * gradient(i)
    # za i-ti koeficijent koji mnozi i-ti atribut
    # gledaju se samo vrednosti i-tog atributa za sve uzorke
    # gradient(i) = (1 / m) * sum(y_predicted - y_target) * features(i)
    # (predicted - self.target) - dimenzije (m x 1)
    # features - dimenzije (m x (n + 1));
    # transponovana matrica ima dimenzije ((n + 1) x m)
    # gradient - dimenzije ((n + 1) x m) x (m x 1) = (n + 1) x 1
        s = self.features.T.dot(predicted - self.target)
        gradient = (1. / len(self.features)) * s
        self.coeff = self.coeff - learning_rate * gradient
        return self.coeff, self.cost()

    def perform_gradient_descent(self, learning_rate, num_iterations=100):

    # Istorija Mean-square error-a kroz iteracije gradijentnog spusta.
        self.mse_history = []
        for i in range(num_iterations):
            _, curr_cost = self.gradient_descent_step(learning_rate)
        self.mse_history.append(curr_cost)
        return self.coeff, self.mse_history

    # features mora biti DataFrame
    def fit(self, features, target):
        self.features = features.copy(deep=True)

    # Pocetna vrednost za koeficijente je 0.
    # self.coeff - dimenzije ((n + 1) x 1)
        coeff_shape = len(features.columns) + 1
        self.coeff = np.zeros(shape=coeff_shape).reshape(-1, 1)
    # Unosi se kolona jedinica za koeficijent c0,
    # kao da je vrednost atributa uz c0 jednaka 1.
        self.features.insert(0, 'c0', np.ones((len(features), 1)))
    # self.features - dimenzije (m x (n + 1))
        self.features = self.features.to_numpy()
    # self.target - dimenzije (m x 1)
        self.target = target.to_numpy().reshape(-1, 1)

spots = 200
d=[]
d.append(np.linspace(0, max(xtrain['gender']), num=spots))
d.append(np.linspace(0, max(xtrain['age']), num=spots))
d.append(np.linspace(0, max(xtrain['annual_salary']), num=spots))
d.append(np.linspace(0, max(xtrain['credit_card_debt']), num=spots))
d.append(np.linspace(0, max(xtrain['net_worth']), num=spots))
estates = pd.DataFrame(data=np.array(d))
# Kreiranje i obucavanje modela
lrgd = LinearRegressionGradientDescent()
lrgd.fit(xtrain, ytrain)
learning_rates = np.array([[0.7],[0.7],[0.7],[0.7],[0.7], [0.0475]])
res_coeff, mse_history = lrgd.perform_gradient_descent(learning_rates, 200)

print(res_coeff)
#print(mse_history[len(mse_history)-1])
lrgd.set_coefficients(res_coeff)
print(f'LRGD MSE: {lrgd.cost():.2f}')
lr_model.coef_ = lrgd.coeff.flatten()[1:]
lr_model.intercept_ = lrgd.coeff.flatten()[0]
print(f'LRGD score: {lr_model.score(xtest, ytest):.2f}')
