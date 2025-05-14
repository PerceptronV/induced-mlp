import numpy as np
import networkx as nx
import seaborn as sns
from matplotlib import pyplot as plt
from utils import permute, brute_force_directionality


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

        return mean_y, max_y, min_y


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
        
        return all_weights, mean_weights, std_weights


class DirectionalityVisualiser(EvalVisualiser):
    def __init__(self, extractor, name='Weights', graphs=False):
        self.extractor = extractor
        self.name = name
        self.graphs = graphs
        self.input_size = None
        self.output_size = None
        self.all_weights = []
    
    def update(self, result):
        weights = self.extractor(result)
        self.input_size = result['model'].input_size
        self.output_size = result['model'].output_size
        self.all_weights.append(weights.detach().cpu().numpy())

    def display(self):
        scores = []
        for w in self.all_weights:
            square = w[:, :-self.input_size]
            directionality, perm = brute_force_directionality(square, self.input_size)
            if self.graphs:
                sns.heatmap(permute(square, perm), annot=True)
                plt.title(f'Directionality of {self.name}: {directionality:.3f}')
                plt.show()
            scores.append(directionality)
        
        mean_dir = np.mean(scores)
        std_dir = np.std(scores)
        max_dir = np.max(scores)
        min_dir = np.min(scores)

        print(f'Directionality: {mean_dir:.3f} $\pm$ {std_dir:.3f}')

        self.scores = scores
        self.mean_dir = mean_dir
        self.std_dir = std_dir
        self.max_dir = max_dir
        self.min_dir = min_dir

        return scores
