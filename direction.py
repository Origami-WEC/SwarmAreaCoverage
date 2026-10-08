from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import numpy as np
from scipy.optimize import minimize

# Dati: X = punti in R^2, y = valori funzione
def get_neighbours_data(neighbours):
    X = np.array([n.position for n in neighbours])
    Y = np.array([n.power for n in neighbours])
    # print(Y)
    return X, Y

def GP_fit(X, Y):   
    # GP fitting
    kernel = RBF(length_scale=3)
    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True)
    gp.fit(X=X, y=Y)
    return gp

def dir(n, res):
    delta = res.x - n.position
    norm = np.linalg.norm(delta)
    if norm == 0:
        return np.array([0, 0])   # Restituisci una direzione di fallback
    return np.divide(delta, norm)

def get_direction(n, neighbours):
    X, Y = get_neighbours_data(neighbours=neighbours)
    try:
        # print(X, Y)
        gp = GP_fit(X, Y)
        def gp_neg(x):
            x = np.array(x).reshape(1, -1)
            y_pred, _ = gp.predict(x, return_std=True)
            return -y_pred[0]
        res = minimize(gp_neg, x0=n.position, bounds=[(n.position[0] - n.vision, n.position[0] + n.vision), 
            (n.position[1] - n.vision, n.position[1] + n.vision)])
    except:
        return normal_direction(n=n)
        
    if res.success:
        return dir(n, res)
    else:
        # Restituisci una direzione predefinita o gestisci l'errore come meglio credi
        return np.array([0, 0])  # Direzione di fallback

def normal_direction(n):
    crowd = n.crowd()
    if len(crowd) == 0 or n.battery < 10:
        targets = n.get_target() # highest power function, i go where the power is higher
        delta = n.space.calculate_difference_vector(n.position, agents=targets)
        delta = delta[0]
        # Normalize direction vector
        norm = np.linalg.norm(delta)
        direction = np.divide(delta, norm)
    elif len(crowd) > 0:
        n.agoraphobic(crowd=crowd)
    else:
        direction = [0, 0]
    #print("direction =", self.direction)
    return direction