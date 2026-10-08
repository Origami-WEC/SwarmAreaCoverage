import numpy as np
from scipy import stats

def estimate_probability(data, lower=None, upper=None):
    """
    Estimate the probability of a value (or range) based on data.

    Parameters:
    - data (list or np.array): The dataset
    - lower (float): Lower bound of the interval (None means -∞)
    - upper (float): Upper bound of the interval (None means +∞)
    - use_kde (bool): Use KDE instead of normal distribution

    Returns:
    - prob (float): Estimated probability
    """

    data = np.array(data)

    # Set bounds
    if lower is None:
        lower = -np.inf
    if upper is None:
        upper = np.inf
    
    mu, std = stats.norm.fit(data)
    prob = np.multiply(stats.norm.cdf(upper, mu, std) - stats.norm.cdf(lower, mu, std), 1)
    #print(prob)

    return prob

def separation_new(self):
    #print(neighbours_power)
    neighbors_power = [self.model.power.get_power(n.position) for n in self.neighbors]

    filtered_separations = [n.separation for n in self.neighbors if np.isfinite(n.separation)]
    if filtered_separations:
        mu, _ = stats.norm.fit(filtered_separations)
    else:
        mu = self.min_separation

    s = 1.25 - estimate_probability(data=neighbors_power, upper=self.model.power.get_power(self.position))
    self.separation = np.multiply(s, mu)
    return

def separation_old(self):
    #print("separation at step ", self.step_number," = ", self.separation)
    self.separation += np.multiply(1 - np.divide(self.battery + 40, 100), 2)

    if self.energy_harvested < self.mean_energy_harvested:
        self.separation = np.multiply(self.min_separation - 1, 3)

    else:
        self.separation  = np.multiply(self.min_separation, 2) - 1
    return

def separation(self):
    if self.info_sep == 'Probabilistic':
        separation_new(self)
    elif self.info_sep == 'Step':
        separation_old(self)

    if self.separation < self.min_separation:
        self.separation = self.min_separation
    if self.separation > 3*self.min_separation:
        self.separation = 3*self.min_separation
    return
