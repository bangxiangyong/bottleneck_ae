import numpy as np
from scipy.stats import binom
import matplotlib.pyplot as plt

from baetorch.baetorch.evaluation import convert_hard_pred

p_threshold = 0.5
p = np.linspace(0, 1.0, 100)
n = 1000
n_anom = 900
conf_func = np.vectorize(lambda p_: 1 - binom.cdf(n, n, p_))
ex_conf = conf_func(p)

test_prob = np.copy(p)
conf_func = np.vectorize(lambda p: 1 - binom.cdf(n - n_anom, n, p))

# BIN
exWise_conf = conf_func(test_prob)

# PROB
# exWise_conf = np.copy(p) ** n

test_hard_pred = convert_hard_pred(test_prob, p_threshold=p_threshold)
np.place(exWise_conf, test_hard_pred == 0, 1 - exWise_conf[test_hard_pred == 0])

plt.figure()
plt.plot(p ** n)

plt.figure()
plt.plot(p, p * (1 - p) * 4)
plt.plot(p, 1 - exWise_conf)


# plt.figure()
# plt.plot(p, ex_conf)


# import scipy.stats as stats
# import matplotlib.pyplot as plt
# import numpy as np
#
# x = np.linspace(0, 20, 100)
# cdf = stats.binom.cdf
# plt.plot(x, cdf(x, 50, 0.2))
# plt.show()
