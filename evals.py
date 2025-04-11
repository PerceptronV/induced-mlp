import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


class EvalVisualiser:
    def update(self):
        ...
    
    def display(self):
        ...


class LineVisualiser(EvalVisualiser):
    def __init__(self, extractor, xlabel=None, ylabel=None, only_values=False):
        self.all_x = []
        self.all_y = []
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.extractor = extractor
        self.only_values = only_values
    
    def update(self, result):
        r = self.extractor(result)
        if self.only_values:
            self.all_y.append(r)
        else:
            self.all_x.append(r[0])
            self.all_y.append(r[1])

    def display(self):
        min_len = min([len(ele) for ele in self.all_y])
        if self.only_values:
            x = np.arange(min_len)
        else:
            x = self.all_x[0][:min_len]
        y = np.array([ele[:min_len] for ele in self.all_y])

        mean_y = np.mean(y, axis=0)
        max_y = np.max(y, axis=0)
        min_y = np.min(y, axis=0)

        plt.plot(x, mean_y, label=f'Mean {self.ylabel}', color='blue')
        plt.fill_between(x, max_y, min_y, color='blue', alpha=0.2, label='Range')

        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.legend()
        plt.tight_layout()
        plt.show()


class WeightVisualiser(EvalVisualiser):
    def __init__(self, extractor, name='Weights', show=None):
        self.show = show
        self.name = name
        self.all_weights = []
        self.extractor = extractor
    
    def update(self, result):
        weight = self.extractor(result)
        self.all_weights.append(weight.detach().cpu().numpy())
    
    def heat(self, matrix, title):
        sns.heatmap(matrix, annot=True)
        plt.title(f'{title} {self.name}')
        plt.tight_layout()
        plt.show()

    def display(self):
        if self.show is None:
            self.show = list(map.keys())
        
        all_weights = np.array(self.all_weights)
        mean_weights = np.mean(all_weights, axis=0)
        std_weights = np.std(all_weights, axis=0)

        map = {
            'mean': (mean_weights, 'Mean'),
            'std': (std_weights, 'Std'),
            'sample': (all_weights[0], 'Sample'),
            'abs-mean': (np.abs(mean_weights), 'Abs Mean'),
            'triu': (np.triu(mean_weights, 1), 'Triu Mean')
        }

        for i in self.show:
            self.heat(*map[i])
