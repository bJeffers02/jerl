import csv
import matplotlib.pyplot as plt
from pathlib import Path

class Plotter:
    def __init__(self):
        self.groups = {
            "Loss": ["total_loss", "actor_loss", "critic_loss"],
            "Reward": ["episode_reward"],
            "Entropy": ["entropy", "entropy_coef", "entropy_bonus"],
            "Time": ["episode_duration", "training_duration", "loss_time"],
        }

        self.keys = [k for group in self.groups.values() for k in group]
        self.data = {k: [] for k in self.keys}
        self.xdata = []

        self.fig, self.axes = plt.subplots(2, 2, figsize=(14, 8))
        self.fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.075, hspace=0.15)

        
        
        color_dict = {
            'episode_reward': 'gold',
            'episode_duration': 'orange',
            'total_loss': 'purple',
            'actor_loss': 'blue',
            'critic_loss': 'red',
            'entropy': 'yellow',
            'entropy_coef': 'lime',
            'entropy_bonus': 'green',
            'training_duration': 'brown',
            'loss_time': 'cyan'
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

    def update_csv(self, output_dir, new_data: dict):
        csv_file = Path(f"{output_dir}/metrics.csv")
        with open(csv_file, 'a', newline='') as csvfile:
            fieldnames = list(new_data.keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if csvfile.tell() == 0:
                writer.writeheader()
            writer.writerow(new_data)

    def run(self):
        self.fig.show()

    def take_screenshots(self, filename="figure.png", output_dir="."):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        full_path = output_path / filename

        self.fig.savefig(full_path, bbox_inches="tight", dpi=300)