import pandas as pd
import numpy as np
from scipy.optimize import minimize

chat_id = 723988166 # Ваш chat ID, не меняйте название переменной

def solution(x: np.array) -> float:
    # Измените код этой функции
    # Это будет вашим решением
    # Не меняйте название функции и её аргументы
    # dev = [i + 1 for i in range(len(x))]
    # return (x / dev).mean()
    
    # Определение функции правдоподобия модели
    def likelihood_func(mu: float, x: np.array) -> float:
        n = len(x)
        return -n * np.log(mu) - np.sum(x) / mu

    # Максимизация функции правдоподобия
    res = minimize(likelihood_func, x0=1, args=(x,), method='Nelder-Mead')
    
    # Возвращение оценки параметра модели
    return res.x[0]
