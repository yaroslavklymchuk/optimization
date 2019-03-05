import numpy as np


def gradient(func, h, params):
    """
    calculates a gradient of a given function by numerical methods
    """
    gradient_vector = np.array([(func(*(params[:i]+[params[i]+h]+params[i+1:]))-
                          func(*(params[:i]+[params[i]-h]+params[i+1:])))/(2*h) for i in range(len(params))])
    return gradient_vector


def gessian(func, h, params):
    """
    calculates a gessian of a given function by numerical methods
    """
    hesse_matrix = np.zeros((len(params), len(params)))
    for i in range(len(params)):
        for j in range(i, len(params)):
            if i==j:
                hesse_matrix[i][j] = (func(*(params[:i] + [params[i]+h] + params[i+1:]))-\
                                      2*func(*params)+func(*(params[:i] + [params[i]-h] + params[i+1:])))/(pow(h,2))
            else:
                hesse_matrix[i][j]=((func(*(params[:i]+[params[i]+h]+params[i+1:j]+[params[j]+h]+params[j+1:]))-\
                                   func(*(params[:i]+[params[i]-h]+params[i+1:j]+[params[j]+h]+params[j+1:])))-\
                                   (func(*(params[:i]+[params[i]+h]+params[i+1:j]+[params[j]-h]+params[j+1:]))-\
                                   func(*(params[:i]+[params[i]-h]+params[i+1:j]+[params[j]-h]+params[j+1:])))
                                   )/(4*pow(h,2))
        for j in range(i):
            hesse_matrix[i][j]=hesse_matrix[j][i]
    
    return hesse_matrix



def gradient_descent_constant_step(func, params, eps, step):
    """
    gradient descent method to minimize a given function with a constant step
    """
    qty_steps=1
    
    dot0 = np.array(params)
    steps = [dot0]
    dot1 = dot0-step*gradient(func, eps, dot0.tolist())
    
    while(np.linalg.norm(dot1-dot0)>eps):
        
        dot0 = dot1

        steps.append(dot0)

        dot1 = dot0 - step*gradient(func, eps, dot0.tolist())
        qty_steps+=1

        print('number of iteration: {}, current point: {}, function value: {}'.format(qty_steps, 
                                                                                      dot1, func(*dot1)))
    
    print("Precision: {}".format(np.linalg.norm(dot1-dot0)))
    
    return steps


def gradient_descent(func, params, eps):
    """
    gradient descent method to minimize a given function
    """
    qty_steps=1
    step=0.001 # step can be chosen in range(0.0001, 0.7)
    
    dot0 = np.array(params)
    steps = [dot0]
    dot1 = dot0-step*gradient(func, eps, dot0.tolist())
    
    while(np.linalg.norm(dot1-dot0)>eps):
        f0 = func(*dot0)
        while(func(*(dot0-step*gradient(func, eps, dot0.tolist())))<f0):
            step*=2
        while(func(*(dot0-step*gradient(func, eps, dot0.tolist())))>f0):
            step*=1/2
        
        dot0 = dot1
        dot1 = dot0-step*gradient(func, eps, dot0.tolist())
        steps.append(dot0)
        
        qty_steps+=1
        
        print('number of iteration: {}, current point: {}, function value: {}'.format(qty_steps,
                                                                                      dot1, func(*dot1)))
    
    print("Precision: {}".format(np.linalg.norm(dot1-dot0)))
    
    return steps



def newton_method(func, params, eps):
    """
    Newton method to minimize a given function
    """
    qty_steps=1
    
    dot0 = np.array(params)
    steps = [dot0]
    dot1 = dot0-np.dot(gradient(func, eps, dot0.tolist()), np.linalg.inv(gessian(func, eps, dot0.tolist())))
    
    while(np.linalg.norm(dot1-dot0)>eps):
        dot0 = dot1

        steps.append(dot0)

        dot1 = dot0 - np.dot(gradient(func, eps, dot0.tolist()), np.linalg.inv(gessian(func, eps, dot0.tolist())))
        qty_steps+=1

        print('number of iteration: {}, current point: {}, function value: {}'.format(qty_steps, 
                                                                                   dot1, func(*dot1)))
    print("Precision: {}".format(np.linalg.norm(dot1-dot0)))
    
    return steps