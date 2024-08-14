import pandas as pd
import numpy as np
import random
from sklearn.datasets import make_regression

X, y = make_regression(n_samples=1000, n_features=14, n_informative=10, noise=15, random_state=42)
X = pd.DataFrame(X)
y = pd.Series(y)
X.columns = [f'col_{col}' for col in X.columns]

class MyLineReg():
    def __init__(self, 
                 n_iter: int = 100, 
                 learning_rate: float = 0.1,  
                 metric: str = None, 
                 reg: str = None, 
                 l1_coef: int = 0, 
                 l2_coef: int = 0,
                 sgd_sample = None,
                 random_state = 42):
        
        self.n_iter  = n_iter
        self.learning_rate = learning_rate
        self.weights = None
        self.metric = metric
        self.mv = 0
        self.reg = reg
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.sgd_sample = sgd_sample
        self.random_state = random_state
    
    def __str__(self) -> str:
        return f"MyLineReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}, {self.mv}"
    
    def count_metric(self,num_rows, y_vector, y_predict) -> None:
        y_mean = (1/num_rows) * np.sum(y_vector)
        if self.metric == 'mae':
            self.mv = (1/num_rows) * np.sum(np.absolute(np.add(y_predict,(-1) * y_vector)))
            
        if self.metric == 'mse':
            self.mv = (1/num_rows) * np.sum(np.square(np.add(y_predict,(-1) * y_vector)))
            
        if self.metric == 'rmse':
            self.mv = np.sqrt((1/num_rows) * np.sum(np.square(np.add(y_predict,(-1)*y_vector))))
            
        if self.metric == 'mape':
            self.mv =(100/num_rows) * np.sum(np.absolute(np.divide(np.add(y_predict,(-1)*y_vector),y_vector)))
        
        if self.metric == 'r2':
            self.mv = 1 - (np.divide(np.sum(np.square(np.add(y_predict,(-1)*y_vector))),np.sum(np.square(np.add(y_mean,(-1)*y_vector)))))
    
    def loss(self,num_rows,y_vector,y_predict,weighted_vector):
        if self.reg == None:
            loss = (1/num_rows) * np.sum(np.square(np.add(y_predict,(-1) * y_vector)))
        
        if self.reg == 'l1':
            loss = (1/num_rows) * np.sum(np.square(np.add(y_predict,(-1)* y_vector))) + self.l1_coef * np.sum(np.absolute(weighted_vector))
        
        if self.reg == 'l2':
            loss = (1/num_rows) * np.sum(np.square(np.add(y_predict,(-1) * y_vector))) + self.l2_coef * np.sum(np.square(weighted_vector))
        
        if self.reg == 'elasticnet':
            loss = (1/num_rows) * np.sum(np.square(np.add(y_predict,(-1) * y_vector))) + self.l2_coef * np.sum(np.square(weighted_vector)) + self.l1_coef * np.sum(np.absolute(weighted_vector))
            
        return loss
    
    def grad(self,num_rows,y_vector,y_predict,X_matrix,weighted_vector):
        if self.reg == None:
            grad = (2/(num_rows)) * np.dot(np.add(y_predict,(-1) * y_vector), X_matrix)
        
        if self.reg == 'l1':
            grad = (2/(num_rows)) * np.dot(np.add(y_predict,(-1) * y_vector), X_matrix) + self.l1_coef * np.sign(weighted_vector)
        
        if self.reg == 'l2':
            grad = (2/(num_rows)) * np.dot(np.add(y_predict,(-1) * y_vector), X_matrix) + self.l2_coef * 2 * weighted_vector
        
        if self.reg == 'elasticnet':
            grad = (2/(num_rows)) * np.dot(np.add(y_predict,(-1) * y_vector), X_matrix) + self.l2_coef * 2 * weighted_vector + self.l1_coef * np.sign(weighted_vector)
        
        return grad
        
        
    def fit(self, X: pd.DataFrame, y: pd.Series, verbose = False) -> None :
        num_rows = X.shape[0]
        num_columns = X.shape[1]
        unit_column = np.ones(num_rows)
        X_matrix = X.to_numpy()
        y_vector = y.to_numpy()
        X_matrix = np.insert(X_matrix,0,unit_column,axis=1)
        weighted_vector = np.ones(num_columns + 1)
        random.seed(self.random_state)
        i = 1
    
        if self.sgd_sample == None: 
            while i <= self.n_iter:
                y_predict = np.dot(X_matrix,weighted_vector)
                self.count_metric( num_rows, y_vector, y_predict)
                loss = self.loss( num_rows,y_vector,y_predict,weighted_vector)
                grad = self.grad(num_rows,y_vector,y_predict,X_matrix,weighted_vector)
                if isinstance(self.learning_rate, float) == True:
                    weighted_vector = np.add(weighted_vector,(-1 * self.learning_rate) * grad)
                else:
                    rate = self.learning_rate(i)
                    weighted_vector = np.add(weighted_vector,(-1 * rate) * grad)
                
                if verbose and i % verbose == 0:
                    print(f'{i} | loss: {loss}| {self.metric} : {self.mv}')
            
                i += 1 
        
        if isinstance(self.sgd_sample, int) == True:
            while i <= self.n_iter:
                sample_rows_idx = random.sample(range(X.shape[0]), self.sgd_sample)
                X_mini_matrix = np.take(X_matrix,sample_rows_idx,axis=0)    
                y_mini_vector = np.take(y_vector,sample_rows_idx)
                y_predict = np.dot(X_matrix, weighted_vector)
                y_mini_predict = np.take(y_predict,sample_rows_idx)
                self.count_metric(num_rows, y_vector, y_predict)
                loss = self.loss(num_rows,y_vector,y_predict,weighted_vector)
                grad = self.grad(num_rows=self.sgd_sample,y_vector=y_mini_vector,y_predict=y_mini_predict, X_matrix=X_mini_matrix, weighted_vector=weighted_vector)                        
                if isinstance(self.learning_rate, float) == True:
                    weighted_vector = np.add(weighted_vector,(-1 * self.learning_rate) * grad)
                else:
                    rate = self.learning_rate(i)
                    weighted_vector = np.add(weighted_vector,(-1 * rate) * grad)
                if verbose and i % verbose == 0:
                    print(f'{i} | loss: {loss}| {self.metric} : {self.mv}')
            
                i += 1 
        
        if isinstance(self.sgd_sample, float) == True:
            while i <= self.n_iter:
                part = round(self.sgd_sample * num_rows)
                sample_rows_idx = random.sample(range(X.shape[0]), part)
                X_mini_matrix = np.take(X_matrix,sample_rows_idx,axis=0)    
                y_mini_vector = np.take(y_vector,sample_rows_idx)
                y_predict = np.dot(X_matrix, weighted_vector)
                y_mini_predict = np.take(y_predict,sample_rows_idx)
                self.count_metric( num_rows, y_vector, y_predict)
                loss = self.loss(num_rows,y_vector,y_predict,weighted_vector)
                grad = self.grad(num_rows=part,y_vector=y_mini_vector,y_predict=y_mini_predict, X_matrix=X_mini_matrix, weighted_vector=weighted_vector)                        
                if isinstance(self.learning_rate, float) == True:
                    weighted_vector = np.add(weighted_vector,(-1 * self.learning_rate) * grad)
                else:
                    rate = self.learning_rate(i)
                    weighted_vector = np.add(weighted_vector,(-1 * rate) * grad)
                if verbose and i % verbose == 0:
                    print(f'{i} | loss: {loss}| {self.metric} : {self.mv}')
            
                i += 1 
        
        self.weights = weighted_vector
        y_predict = np.dot(X_matrix,self.weights)
        self.count_metric( num_rows, y_vector, y_predict)
    
    def get_coef(self) -> None:
        return self.weights[1:]
    
    def predict(self,X) -> np.array:
        num_rows = X.shape[0]
        unit_column = np.ones(num_rows)
        X.insert(loc= 0, column='fake_dev' ,value=unit_column)
        X_matrix = X.to_numpy()
        y_predict = np.matmul(X_matrix,self.weights)
        return y_predict
    
    def get_best_score(self) -> float:
        return self.mv
    
    
obj = MyLineReg(sgd_sample=1,reg= 'l2', l2_coef=1)

obj.fit(X,y,verbose = True)