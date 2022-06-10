import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns

# y = np.array([0, 0, 1, 1])
# y_classes = np.concatenate((np.zeros(len(normal_scores)), np.ones(len(anomaly_scores))))
# scores = np.array([0.1, 0.4, 0.35, 0.8])
# anomaly_scores = np.random.uniform(-0.5, 3, 100)
# anomaly_scores = np.random.uniform(-1.5, 1.5, 100)
np.random.seed(98)
n_samples = 100
strong_normal_scores = np.random.uniform(-1, 1, n_samples)
weak_normal_scores = np.random.uniform(-1, 1, n_samples)

strong_anomaly_scores = np.random.uniform(0.5, 2.5, n_samples)
weak_anomaly_scores = np.random.uniform(-1.25, 1.1, n_samples)

base_color = "black"
inlier_color = "tab:blue"
strong_color = "tab:green"
weak_color = "tab:orange"


def roc_curve_wrap(normal_scores, anomaly_scores):
    fpr, tpr, thresholds = metrics.roc_curve(
        np.concatenate((np.zeros(len(normal_scores)), np.ones(len(anomaly_scores)))),
        np.concatenate((normal_scores, anomaly_scores)),
    )

    return fpr, tpr, thresholds, metrics.auc(fpr, tpr)


strong_fpr, strong_tpr, thresholds, strong_auroc = roc_curve_wrap(
    strong_normal_scores, strong_anomaly_scores
)
weak_fpr, weak_tpr, thresholds, weak_auroc = roc_curve_wrap(
    weak_normal_scores, weak_anomaly_scores
)

# ========
alpha = 0.125
density = True
# figsize = (6.25, 3)
figsize = (10, 3.25)
kde_adjust = 0.65
hist_legend_pos = "lower center"

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

# PLOT KDE
sns.kdeplot(data=weak_normal_scores, ax=ax1, color=inlier_color)
sns.kdeplot(data=weak_anomaly_scores, ax=ax1, color=weak_color, bw_adjust=kde_adjust)

sns.kdeplot(data=strong_normal_scores, ax=ax2, color=inlier_color)
sns.kdeplot(
    data=strong_anomaly_scores, ax=ax2, color=strong_color, bw_adjust=kde_adjust
)

# HISTOGRAM (strong)
ax1.hist(weak_normal_scores, alpha=alpha, density=density, color=inlier_color)
ax1.hist(weak_anomaly_scores, alpha=alpha, density=density, color=weak_color)
ax2.hist(strong_normal_scores, alpha=alpha, density=density, color=inlier_color)
ax2.hist(strong_anomaly_scores, alpha=alpha, density=density, color=strong_color)

# PLOT AUROC
ax3.plot(np.linspace(0, 1, 20), np.linspace(0, 1, 20), "--", color=base_color)
ax3.plot(strong_fpr, strong_tpr, color=strong_color)
ax3.plot(weak_fpr, weak_tpr, color=weak_color)


# LEGENDS
weak_auroc_legend = "Weak detector" + "(" + str(weak_auroc.round(2)) + ")"
strong_auroc_legend = "Strong detector" + "(" + str(strong_auroc.round(2)) + ")"

ax1.legend(["Inliers", "Anomalies"], loc=hist_legend_pos)
ax2.legend(["Inliers", "Anomalies "], loc=hist_legend_pos)
ax3.legend(["Baseline (0.50)", weak_auroc_legend, strong_auroc_legend])

# SET LABELS
ax1.set_xlabel("Anomaly scores (Weak detector)")
ax2.set_xlabel("Anomaly scores (Strong detector)")

ax3.set_ylabel("True positive rate (sensitivity)")
ax3.set_xlabel("False positive rate (1-specificity)")

# TITLES
ax1.set_title("(a) Weak detector")
ax2.set_title("(b) Strong detector")
ax3.set_title("(c) ROC curves")

# SAVE FIG
fig.tight_layout()
fig.savefig("auroc_example.png", dpi=500)
