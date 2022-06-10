import matplotlib.pyplot as plt

def plot(fs,dpi):
    fig, ax=plt.subplots(figsize=fs, dpi=dpi)
    ax.set_title("Figsize: {}, dpi: {}".format(fs,dpi))
    ax.plot([2,4,1,5], label="Label")
    ax.legend()

figsize=(4,2)
for i in range(2,6):
    plot(figsize, i*100)
    plt.tight_layout()

