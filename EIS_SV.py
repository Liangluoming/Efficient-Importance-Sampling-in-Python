import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import time
from scipy.stats import norm


def SV_generator(N, T, v, phi, sigma_square):
    """
    对数正态分布的SV模型生成器
    $$
    y_t \sim N(0, exp(h_t)), h_t \sim N(v + \phi h_{t-1}, \sigma^2), h_0 \sim N(0, \sigma^2)
    $$

    Args:
        N : number of particles
        T : times
        v : 漂移项
        phi : 相关系数
        sigma_square: 状态变量方差

    Returns:
        observations: 观测值
        states: 状态变量
    """
    sigma = np.sqrt(sigma_square)
    states = np.zeros((N, T))
    for t in range(T):
        np.random.seed(t)
    
        if t == 0:
            ## 第一期
            state = 0 + sigma * np.random.normal(loc = 0, scale = 1, size = N)
            states[:, t] = state
        else:
            state = v + phi * states[:, t - 1] + sigma * np.random.normal(loc = 0, scale = 1, size = N)
            states[:, t] = state
    ## 由状态变量生成观测值
    np.random.seed(T)
    observations = np.sqrt(np.exp(states)) * np.random.normal(loc = 0, scale = 1, size = (N, T))
    print("样本生成成功！")
    return states, observations

def CRN(states, states_minus_1, a, v, phi, sigma_square, seed = 123):
    """
    CRN生成proposal函数

    Args:
        states: 当前的状态变量
        states_minus_1: 上一期的状态变量

    Return:
        服从proposal分布的particles
    """

    N = len(states)
    np.random.seed(seed)

    propsal_mean = (v + phi * states_minus_1) / (1 + a * sigma_square)
    propsal_var = sigma_square / (1 + a * sigma_square)

    return propsal_mean + np.sqrt(propsal_var) * np.random.normal(loc = 0, scale = 1, size = N)



def EIS(observations, states, v, phi, sigma_square, max_iter = 1000, epsilon = 0.01):
    """
    针对上述SV模型的EIS估计
    选用的 kernel function: $$p(h_t | h_{t-1}) e^{a * h_{t-1}^2 / 2 } $$  chi function: $$ \sqrt{2 * pi * sigma^2} e^{a * h_{t-1}^2 / 2 }$$

    Args:
        observations: 观测值
        states: 状态变量
        v : 漂移项
        phi : 相关系数
        sigma_square: 状态变量方差
        max_iter: 最大迭代次数
        epsilon: 收敛判断阈值
    Returns:
        log_maxlikelihood: 对数最大似然
    """
    N , T = observations.shape
    obs_sigma_square = np.exp(states) ## 观测值方差
    start_time = time.time()
    for iter in range(max_iter):
        iter_params = np.zeros((T, 2))
        for t in range(1, T):
            ## 观测pdf的对数
            ln_obs = - 0.5 * (np.log(2 * np.pi * obs_sigma_square[:, T - t]) + observations[:, T - t] ** 2 / obs_sigma_square[:, T - t])
            ## 转移函数的对数
            ln_transition = - 0.5 * (np.log(2 * np.pi * sigma_square) + (states[:, T - t] - v - phi * states[:, T - t - 1]) ** 2 / sigma_square)
            y = ln_obs + ln_transition
            if t > 1:
                ln_chi = - 0.5 * (np.log(1 + iter_params[T - t + 1, 0] * sigma_square) + (iter_params[T - t + 1, 0] * (v + phi * states[:, T - t - 1]) ** 2 / (1 + iter_params[T - t + 1, 0] * sigma_square)))
                y += ln_chi

            x = - 0.5 * states[:, T - t] ** 2
            y = np.reshape(y, (-1, 1))
            x = np.reshape(x, (-1, 1))

            lm_model = LinearRegression()
            lm_model.fit(x, y)
            iter_params[T - t, 0] = lm_model.coef_[0][0]
            iter_params[T - t, 1] = lm_model.intercept_[0]

        ## 第1期对数最大似然

        ln_obs = - 0.5 * (np.log(2 * np.pi * obs_sigma_square[:, 0]) + observations[:, 0] ** 2 / obs_sigma_square[:, 0])
        ln_transition = - 0.5 *(np.log(2 * np.pi * sigma_square) + (states[:, 0] ** 2 / sigma_square))
        ln_chi = - 0.5 * (np.log(1 + iter_params[1, 0] * sigma_square) + (iter_params[1, 0] * (v + phi * states[:, 0]) ** 2 / (1 + iter_params[1, 0] * sigma_square)))
        y = ln_obs + ln_transition + ln_chi
        x = - 0.5 * states[:, 0] ** 2
        y = np.reshape(y, (-1, 1))
        x = np.reshape(x, (-1, 1))
        lm_model = LinearRegression()
        lm_model.fit(x, y)
        iter_params[0, 0] = lm_model.coef_[0][0]
        iter_params[0, 1] = lm_model.intercept_[0]


        if iter == 0:
            params = iter_params.copy()
        else:
            if np.mean(np.linalg.norm(params - iter_params)) < epsilon:
                end_time = time.time()
                print("所需时间:{}s".format(round(end_time - start_time, 2)))
                print("达到收敛的迭代次数:{}".format(iter))
                params = iter_params.copy()
                break
            else:
                params = iter_params.copy()

        ## 依据新参数重新生成服从proposal分布particles
        for t in range(T):
            if t == 0:
                np.random.seed(t)
                states[:, 0] = np.sqrt(sigma_square / (1 + params[t, 0] * sigma_square)) * np.random.normal(loc = 0, scale = 1, size = N)
            else:
                states[:, t] = CRN(states[:, t], states[:, t - 1], params[t, 0], v, phi, sigma_square, seed = t)
    ## 计算对数极大似然
    ln_obs = - 0.5 * (np.log(2 * np.pi * obs_sigma_square) + observations ** 2 / obs_sigma_square) 
    ln_transition_0 = - 0.5 *(np.log(2 * np.pi * sigma_square) + (states[:, 0] ** 2 / sigma_square))
    ln_propsal_0 = - 0.5 * (np.log(2 * np.pi * sigma_square / (1 + params[0, 0] * sigma_square)) + (states[:, 0] ** 2 / (1 + params[0, 0] * sigma_square)))
    ln_transition_1 = - 0.5 * (np.log(2 * np.pi * sigma_square) + (states[:, 1:] - v - phi * states[:, : -1]) ** 2 / sigma_square)
    ln_propsal_1 = - 0.5 * (np.log(2 * np.pi * sigma_square / (1 + params[1:, 0] * sigma_square)) + ((states[:, 1:] - v - phi * states[:, : -1]) ** 2 / (1 + params[1:, 0] * sigma_square)))
    
    ln_transition_0 = np.reshape(ln_transition_0, (-1, 1))
    ln_propsal_0 = np.reshape(ln_propsal_0, (-1, 1))
    ln_transition = np.column_stack((ln_transition_0, ln_transition_1))
    ln_propsal = np.column_stack((ln_propsal_0, ln_propsal_1))
    L = ln_obs + ln_transition - ln_propsal
    L = np.sum(L, axis = 1)
    log_maxlikelihood = np.mean(L, axis = 0)
    return log_maxlikelihood

phi = 0.9
v = 0.1
sigma_square = 1
N = 1000
T = 500

states, observations = SV_generator(N, T, v, phi, sigma_square)
maxlikelihood = EIS(observations, states, v, phi, sigma_square, max_iter = 1000, epsilon = 0.01)
print(maxlikelihood)


# phi = 0.9
# v = 0.1
# sigma_square = 1
# N = 10000
# T = 5000

# states, observations = SV_generator(N, T, v, phi, sigma_square)
# maxlikelihood = EIS(observations, states, v, phi, sigma_square, max_iter = 1000, epsilon = 0.01)
# print(maxlikelihood)