import numpy as np


class sigmoid:
    def __init__(self, x):
        self.x = x

    def function(self):
        return 1/(1+np.exp(-self.x))

    def derivative(self):
        return self.function()*(1-self.function())


class tanh:
    def __init__(self, x):
        self.x = x

    def function(self):
        return 2/(1+np.exp(-2*self.x))

    def derivative(self):
        return 1-np.square(self.function())


class ArcTen:
    def __init__(self, x):
        self.x = x

    def function(self):
        return np.arctan(self.x)

    def derivative(self):
        return 1/(np.square(self.x)+1)


class Bent_identity:
    def __init__(self, x):
        self.x = x

    def function(self):
        return (np.square(np.sqrt(self.x)+1)-1)/2+self.x

    def derivative(self):
        return self.x/(2*np.sqrt(np.square(self.x)+1))+1


class Binary_step:
    def __init__(self, x):
        self.x = x

    def funtion(self):
        if self.x < 0: return 0
        elif self.x >= 0: return 1

    def derivative(self):
        return np.zeros(self.x.shape)


class Exp_Linear_unit:
    def __init__(self, x, alfa):
        self.x = x
        self.alfa = alfa

    def function(self):
        if self.x < 0: return self.alfa*(np.exp(self.x)-1)
        elif self.x >= 0: return self.x

    def derivative(self):
        if self.x < 0: return self.alfa*np.exp(self.x)
        elif self.x >= 0: return np.ones(self.x.shape)


class Gaussian:
    def __init__(self, x):
        self.x = x

    def function(self):
        return np.exp(-np.square(self.x))

    def derivative(self):
        return -2*self.x*self.function()


class identity:
    def __init__(self, x):
        self.x = x

    def function(self):
        return self.x

    def derivative(self):
        return np.ones(self.x.shape)


# Parameteric Rectified Linear Unit
class PReLU:
    def __init__(self, x, alfa):
        self.x = x
        self.alfa = alfa

    def function(self):
        if self.x < 0: return self.x*self.alfa
        elif self.x >= 0: return self.x

    def derivative(self):
        if self.x < 0: return self.alfa
        elif self.x >= 0: return np.ones(self.x.shape)


class Sinc:
    def __init__(self, x):
        self.x = x

    def function(self):
        if self.x == 0: return np.ones(self.x.shape)
        elif self.x != 0: return np.sin(self.x)/self.x

    def dervative(self):
        if self.x == 0: return np.zeros(self.x.shape)
        elif self.x != 0: return np.cos(self.x)/self.x-np.sin(self.x)/np.square(self.x)


class Sinusoid:
    def __init__(self, x):
        self.x = x

    def function(self):
        return np.sin(self.x)

    def derivative(self):
        return np.cos(self.x)


class SoftExponential:
    def __init__(self, x, alfa):
        self.x = x
        self.alfa = alfa

    def function(self):
        if self.alfa < 0: return -np.log(1-self.alfa*(self.x+self.alfa))/self.alfa
        elif self.alfa == 0:return self.x
        elif self.alfa > 0: return (np.exp(self.alfa*self.x)-1)/self.alfa

    def derivative(self):
        if self.alfa < 0:return 1/(1-self.alfa*(self.alfa+self.x))
        elif self.alfa >= 0:return np.exp(self.alfa*self.x)


class SoftPlus:
    def __init__(self, x):
        self.x = x

    def function(self):
        return np.log(1+np.exp(self.x))

    def derivative(self):
        return 1/(1+np.exp(-self.x))


class Softsign:
    def __init__(self, x):
        self.x = x

    def function(self):
        return self.x/(1+np.abs(self.x))

    def derivative(self):
        return 1/np.square(1+np.abs(self.x))


class Tanh:
    def __init__(self, x):
        self.x = x

    def function(self):
        return 2/(1+np.exp(-2*self.x))-1

    def derivative(self):
        return 1-np.square(self.function())

