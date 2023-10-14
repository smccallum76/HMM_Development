"""
This simulator is super basic and intended to generate a string of neutral and positive events. For now this
will be set up as 0's and 1', where a vast majority of the events are 0's:
What is needed?
- Coin_1 --> Fires off 99% of the time in the range N(mu=3, sd=2)...or whatever
- Coin_2 --> Fires off 1% of the time in the range N(mu=5, sd=1)...or whatever

Using the simple_sim output, I need to:
- Use a HMM to determine if coin_1 (neutral) or coin_2 (sweep) was used
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

samples = 10000
cut = 0.95
data = pd.DataFrame(columns=['state', 'label', 'value'])

for i in range(samples):
    select = np.random.uniform(low=0, high=1, size=1)[0]
    if select <= cut:
        coin = np.random.normal(loc=3, scale=2, size=1)
        state = 0
        label = 'N'

    else:
        coin = np.random.normal(loc=8, scale=2, size=1)
        state = 1
        label = 'P'
    data.loc[i, 'state'] = state
    data.loc[i, 'label'] = label
    data.loc[i, 'value'] = coin
    data.loc[i, 'snp'] = i+1

neutral = data[data['state'] == 0]
positive = data[data['state'] == 1]

print("Count of Neutral SNPs: ", len(neutral))
print("Count of Positive SNPs: ", len(positive))

data.to_csv('output/simple_sims.csv', index=False)
"""
Next steps:
1. For now we will assume the distributions are known as the first objective is to recreate the N,P string using HMM
2. Therefore, import the output from this module into a separate file that performs HMM using python
3. Then write HMM code in a fashion similar to the final (but used packages where possible)
"""

# check the data


# plt.figure()
# plt.hist(neutral['value'], bins=50, color='red', alpha=0.5)
# plt.hist(positive['value'], bins=50, color='blue', alpha=0.5)
# plt.show()

# plt.figure()
# plt.bar(x=neutral['snp'], height=neutral['state'], color='red')
# plt.bar(x=positive['snp'], height=positive['state'], color='blue')
# plt.show()
