from abc import ABC, abstractmethod
import math
import numpy as np
import statistics
import matplotlib.pyplot as plt
import random
from scipy import stats


# Итерфейс для любой случайной величины
class RandomVariable(ABC):
    @abstractmethod
    def pdf(self, x):
        pass

    @abstractmethod
    def cdf(self, x):
        pass

    @abstractmethod
    def quantile(self, alpha):
        pass


 # Равномерное распределение
class UniformRandomVariable(RandomVariable):
  def __init__(self, location=0, scale=1) -> None:
    super().__init__()
    self.location = location
    self.scale = scale

  def pdf(self, x):
    if x >= self.location and x <= self.scale:
      return 1 / (self.scale - self.location)
    else:
      return 0

  def cdf(self, x):
    if x <= self.location:
      return 0
    elif x >= self.scale:
      return 1
    else:
      return (x - self.location) / (self.scale - self.location)

  def quantile(self, alpha):
    return self.location + alpha * (self.scale - self.location)


# Непараметрическая случайная величина
class NonParametricRandomVariable(RandomVariable):
    def __init__(self, source_sample) -> None:
        super().__init__()
        self.source_sample = sorted(source_sample)

    def pdf(self, x):
        if x in self.source_sample:
            return float('inf')
        return 0

    @staticmethod
    def heaviside_function(x):
        if x > 0:
            return 1
        else:
            return 0

    def cdf(self, x):
        return np.mean(np.vectorize(self.heaviside_function)(x - self.source_sample))

    def quantile(self, alpha):
        index = int(alpha * len(self.source_sample))
        return self.source_sample[index]


# Интерфейс для генератора псевдослучайных величин
class RandomNumberGenerator(ABC):
    def __init__(self, random_variable: RandomVariable):
        self.random_variable = random_variable

    @abstractmethod
    def get(self, N):
        pass


# Генератор случайных выбросов Тьюки
class TukeyRandomNumberGenerator(RandomNumberGenerator):
    def __init__(self, random_variable: RandomVariable, laplaceRandomVariable: RandomVariable, epsilon):
        super().__init__(random_variable)
        self.epsilon = epsilon
        self.laplaceRandomVariable = laplaceRandomVariable

    def get(self, N):
        sample = []
        us = np.random.uniform(0, 1, N)
        for x in us:
            if x < self.epsilon:
                sample.append(self.laplaceRandomVariable.quantile(random.random()))
            else:
                sample.append(self.random_variable.quantile(random.random()))
        return sample


# Генератор псевдослучайных величин
class SimpleRandomNumberGenerator(RandomNumberGenerator):
    def __init__(self, random_variable: RandomVariable):
        super().__init__(random_variable)

    def get(self, N):
        us = np.random.uniform(0, 1, N)
        return np.vectorize(self.random_variable.quantile)(us)


# Оценки
class Estimation(ABC):
    @abstractmethod
    def estimate(self, sample):
        pass


# Оценка усеченное среднее 1
class YsechMean1(Estimation):
  def estimate(self, sample):
    return stats.trim_mean(sample, 0.05)


# Оценка усеченное среднее 2
class YsechMean2(Estimation):
  def estimate(self, sample):
    return stats.trim_mean(sample, 0.2)


class Mean(Estimation):
    def estimate(self, sample):
        return statistics.mean(sample)


class Var(Estimation):
    def estimate(self, sample):
        return statistics.variance(sample)


# Моделинг
class Modelling(ABC):
    def __init__(self, gen: RandomNumberGenerator, estimations: list, M: int, truth_value: float):
        self.gen = gen
        self.estimations = estimations
        self.M = M
        self.truth_value = truth_value

        # Выборки оценок
        self.estimations_sample = np.zeros((self.M, len(self.estimations)), dtype=np.float64)

    # Метод, оценивающий квадрат смещения оценок
    def estimate_bias_sqr(self):
        b = np.array([(Mean().estimate(self.estimations_sample[:, i]) - 0.5) ** 2 for i in range(len(self.estimations))])
        print('b2', b)
        return b

    # Метод, оценивающий дисперсию оценок
    def estimate_var(self):
        a = np.array([Var().estimate(self.estimations_sample[:, i]) for i in range(len(self.estimations))])
        print('Дисперсия', a)
        return a

    # Метод, оценивающий СКО оценок
    def estimate_mse(self):
        return self.estimate_bias_sqr() + self.estimate_var()

    def get_samples(self):
        return self.estimations_sample

    def get_sample(self):
        return self.gen.get(N)

    def run(self):
        for i in range(self.M):
            sample = self.get_sample()
            self.estimations_sample[i, :] = [e.estimate(sample) for e in self.estimations]


# Сглаженная случайная величина
class SmoothedRandomVariable(RandomVariable):
    @staticmethod
    def _k(x):
        if abs(x) <= 1:
            return 0.75 * (1 - x * x)
        else:
            return 0

    @staticmethod
    def _K(x):
        if x < -1:
            return 0
        elif -1 <= x < 1:
            return 0.5 + 0.75 * (x - x ** 3 / 3)
        else:
            return 1

    def __init__(self, sample, h):
        self.sample = sample
        self.h = h

    def pdf(self, x):
        return np.mean([SmoothedRandomVariable._k((x - y) / self.h) for y in self.sample]) / self.h

    def cdf(self, x):
        return np.mean([SmoothedRandomVariable._K((x - y) / self.h) for y in self.sample])

    def quantile(self, alpha):
        raise NotImplementedError


location = int(input("location: "))
scale = float(input("scale: "))
N = int(input("Объём выборки: "))
M = int(input("Количество ревыборок: "))


rv = UniformRandomVariable(location, scale)
generator = SimpleRandomNumberGenerator(rv)
sample = generator.get(N)
rv1 = NonParametricRandomVariable(sample)
generator1 = SimpleRandomNumberGenerator(rv1)


# Симметричные выбросы
erv = UniformRandomVariable(location, scale + 2)
generator2 = TukeyRandomNumberGenerator(rv, erv, 0.1)
sample2 = generator2.get(N)
rv2 = NonParametricRandomVariable(sample2)
generator3 = SimpleRandomNumberGenerator(rv2)


# Асимметричные выбросы
erv = UniformRandomVariable(location + 5, scale)
generator4 = TukeyRandomNumberGenerator(rv, erv, 0.1)
sample2 = generator2.get(N)
rv2 = NonParametricRandomVariable(sample2)
generator5 = SimpleRandomNumberGenerator(rv2)

modelling = Modelling(generator5, [ YsechMean1(), YsechMean2() ], M, location)
modelling.run()
estimate_mse = modelling.estimate_mse()
print(estimate_mse)
print(f'Оценка1/оценка2 {estimate_mse[0] / estimate_mse[1]}')
print(f'Оценка2/оценка1 {estimate_mse[1] / estimate_mse[0]}')
print()

#bandwidth = 0.01
bandwidth = float(input('Параметр размытости: '))

samples = modelling.get_samples()
POINTS = 100

for i in range(samples.shape[1]):
    sample = samples[:, i]
    X_min = min(sample)
    X_max = max(sample)
    x = np.linspace(X_min, X_max, POINTS)
    srv = SmoothedRandomVariable(sample, bandwidth)
    y = np.vectorize(srv.pdf)(x)
    plt.plot(x, y)

plt.show()
