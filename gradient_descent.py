import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets.samples_generator import make_regression

def gradient_descent(x,y,alpha,ep,max_iter):
    converged=False
    t0=1
    t1=2
    m=x.shape[0]
    iter=0

    J = sum([(t0 + t1*x[i] - y[i])**2 for i in range(m)])

    while not converged:
        grad0=(1/m)*(sum([(t0 + t1*x[i] - y[i]) for i in range(m)]))
        grad1=(1/m)*(sum([(t0 + t1*x[i] - y[i])*x[i] for i in range(m)]))

        temp0=t0-alpha*grad0
        temp1=t1-alpha*grad1

        t0=temp0
        t1=temp1

        e = sum( [ (t0 + t1*x[i] - y[i])**2 for i in range(m)] )

        if(J-e<ep):
            converged=True

        J=e
        iter+=1

        if(iter>=max_iter):
            converged=True

    return t0,t1


if(__name__=="__main__"):
    x, y = make_regression(n_samples=200, n_features=1, n_informative=1,random_state=0, noise=200)

    alpha=0.01
    ep=0.001
    max_iter=3000
    theta0, theta1 = gradient_descent(x, y, alpha, ep, max_iter)

    y_predict=[(theta0+theta1*x[i]) for i in range(x.shape[0])]

    plt.plot(x,y,"o")
    plt.plot(x,y_predict,"k-")
    plt.show()
