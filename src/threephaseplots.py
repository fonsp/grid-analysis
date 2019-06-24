import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0, 2*np.pi, num=1000)

def get_phase_currents(num_phases):
    tmat = np.ones((num_phases, 1)) @ t[np.newaxis, :]
    offsets = np.linspace(0, 2*np.pi, num=num_phases, endpoint=False)

    tmat += offsets[:, np.newaxis] @ np.ones((1, 1000))

    return np.sin(tmat)

n = 3
currents = get_phase_currents(n)
currents_squared = currents * currents

sum_i_squared = (np.ones((1, n)) @ currents_squared).flatten()

fig, ax = plt.subplots()

for i, current in enumerate(currents):
    ax.plot(t, current, label="Phase {}".format(i+1))

ax.plot(t, sum_i_squared, label="Sum of squared currents")
