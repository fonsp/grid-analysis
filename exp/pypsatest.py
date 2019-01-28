#%%
import numpy as np
import matplotlib.pyplot as plt

#%%
x = np.linspace(0,5,100)
x += 1
y = np.sin(x)

print(x)

#%%
plt.plot(x,y)
plt.show()
