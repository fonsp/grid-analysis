import numpy as np
import matplotlib.pyplot as plt

num = 1000
t = np.linspace(0, 2*np.pi, num=num)

fullorhalf = True

def get_phase_currents(num_phases):
    tmat = np.ones((num_phases, 1)) @ t[np.newaxis, :]
    mult = 2 if fullorhalf else 1
    offsets = np.linspace(0, -np.pi * mult, num=num_phases, endpoint=False)

    tmat += offsets[:, np.newaxis] @ np.ones((1, num))

    return np.sin(tmat)


fig, ax = plt.subplots(3, 3, sharex=True, figsize=(12, 6))

ax[0, 0].set_ylabel("Current (A)")
ax[1, 0].set_ylabel("Transmitted power (W)")
ax[2, 0].set_ylabel("Return current (A)")

for n in range(1, 4):
    currents = get_phase_currents(n)
    currents_squared = currents * currents

    sum_i_squared = (np.ones((1, n)) @ currents_squared).flatten()

    total_energy = np.mean(sum_i_squared)

    ax[1, n-1].set_xticks(np.arange(0, 40, 2))

    for i, (color, current) in enumerate(zip(["brown", "black", "grey"], currents)):
        ax[0, n-1].plot((1e3*t / 50) / 2.0 / np.pi, current / np.sqrt(total_energy), label="Phase {}".format(i+1), color=color)

    ax[1, n-1].plot((1e3*t / 50) / 2.0 / np.pi, sum_i_squared / total_energy, label="Sum of squared currents", color="red")
    ax[2, n-1].plot((1e3*t / 50) / 2.0 / np.pi, -(np.ones((1, n)) @ currents).flatten()  / np.sqrt(total_energy), label="Sum of squared currents", color="blue")

    ax[1, n-1].set_xlabel("Time (ms)")
    ax[0, n-1].legend(loc="lower left")

    ax[0, n-1].grid()
    ax[1, n-1].grid()
    ax[2, n-1].grid()

    ax[1, n-1].set_ylim(0,2.3)
    ax[2, n-1].set_ylim(-2,2)

plt.savefig("3phasesfull.pdf" if fullorhalf else "3phaseshalf.pdf")
