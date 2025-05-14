import matplotlib.pyplot as plt
from pathlib import Path

class Plotter:
    def __init__(self):
        self.groups = {
            "Loss": ["total_loss", "actor_loss", "critic_loss"],
            "Reward": ["episode_reward"],
            "Entropy": ["entropy", "entropy_coef"],
            "Time": ["episode_duration", "training_duration"],
        }

        self.keys = [k for group in self.groups.values() for k in group]
        self.data = {k: [] for k in self.keys}
        self.xdata = []

        self.fig, self.axes = plt.subplots(2, 2, figsize=(14, 8))
        self.fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.075, hspace=0.15)

        
        
        color_dict = {
            'episode_reward': (0.12156862745098039, 0.4666666666666667, 0.7058823529411765, 1.0),
            'episode_duration': (1.0, 0.4980392156862745, 0.054901960784313725, 1.0),
            'total_loss': (0.17254901960784313, 0.6274509803921569, 0.17254901960784313, 1.0),
            'actor_loss': (0.8392156862745098, 0.15294117647058825, 0.1568627450980392, 1.0),
            'critic_loss': (0.5490196078431373, 0.33725490196078434, 0.29411764705882354, 1.0),
            'entropy': (0.8901960784313725, 0.4666666666666667, 0.7607843137254902, 1.0),
            'entropy_coef': (0.4980392156862745, 0.4980392156862745, 0.4980392156862745, 1.0),
            'training_duration': (0.7372549019607844, 0.7411764705882353, 0.13333333333333333, 1.0)
        }
        self.lines = {}
        for ax, (group_title, keys) in zip(self.axes.flat, self.groups.items()):
            
            for key in keys:
                line, = ax.plot([], [], label=key, color=color_dict[key])
                self.lines[key] = line

            ax.grid(True)
            ax.legend(loc="lower left", fontsize=8, frameon=True, borderpad=0.5, labelspacing=0.5, facecolor='white')
            ax.tick_params(labelbottom=True, labelleft=True, labelsize=8)
            ax.set_title(group_title, fontsize=12)

        plt.ion()

    def update_data(self, new_data: dict):
        if not self.xdata:
            self.xdata.append(0)
        else:
            self.xdata.append(self.xdata[-1] + 1)

        for key in self.keys:
            self.data[key].append(new_data.get(key, 0))

            line = self.lines[key]
            line.set_data(self.xdata, self.data[key])
            ax = line.axes
            ax.relim()
            ax.autoscale_view()

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def run(self):
        self.fig.show()

    def take_screenshots(self, filename="figure.png", output_dir="."):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        full_path = output_path / filename

        self.fig.savefig(full_path, bbox_inches="tight", dpi=300)