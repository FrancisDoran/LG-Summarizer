import os

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('QtAgg')


"""
matplotlib integration

expected input data:
    * baseline model summary
    * custom model summary
    * reference summary
    * reference article

Independent Variables:
    * model type (baseline, custom)

Dependent Variables:
    * rouge-1, rouge-2, rouge-len
        * precision, recall, f1

        (FOR EACH)


So we can produce 3 bar charts with 6 bars each.

Where each chart represents one of the ROUGE metrics (1, 2, L)...

And each bar represents one of the precision, recall, f1 scores for the two models (baseline and custom).
"""

"""
Empty template for metric dictionary
"""
_TEMP_METRIC_DICT = {
    "baseline": {
        "rouge-1": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
        "rouge-2": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
        "rouge-len": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
    },
    "custom": {
        "rouge-1": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
        "rouge-2": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
        "rouge-len": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
    },
    "reference_summary": "",
    "reference_article": "",
}


"""
MetricDictionary
"""
class MetricDictionary:

    data: dict

    def __init__(self):
        #initialize the metric dictionary using a copy of the dictionary template
        import copy
        self.data = copy.deepcopy(_TEMP_METRIC_DICT)
 
    def add_metric(self, model_type: str, metric_name: str, data_point_name: str, value: float):
        if model_type not in self.data:
            print("Incorrect model type, must be 'baseline' or 'custom'")
            return

        if metric_name not in self.data[model_type]:
            print("Incorrect metric name, must be 'rouge-1', 'rouge-2', or 'rouge-len'")
            return

        if data_point_name not in self.data[model_type][metric_name]:
            print("Incorrect data point, must be 'precision', 'recall', or 'f1'")
            return

        self.data[model_type][metric_name][data_point_name] = value
        return self

    def get(self):
        return self.data


example_data = {
        "baseline": {
            "rouge-1": {"precision": 0.5, "recall": 0.4, "f1": 0.45},
            "rouge-2": {"precision": 0.3, "recall": 0.2, "f1": 0.25},
            "rouge-len": {"precision": 0.4, "recall": 0.3, "f1": 0.35},
        },
        "custom": {
            "rouge-1": {"precision": 0.6, "recall": 0.5, "f1": 0.55},
            "rouge-2": {"precision": 0.4, "recall": 0.3, "f1": 0.35},
            "rouge-len": {"precision": 0.5, "recall": 0.4, "f1": 0.45},
        },
        "reference_summary": "This is the reference summary.",
        "reference_article": "This is the reference article.",
}


def create_bar_graph(data=example_data):
    metrics = ["rouge-1", "rouge-2", "rouge-len"]
    scores = ["precision", "recall", "f1"]

    os.makedirs('generated_plots', exist_ok=True)

    for metric in metrics:
        plt.figure(figsize=(10, 6))
        baseline_scores = [data["baseline"][metric][score] for score in scores]
        custom_scores = [data["custom"][metric][score] for score in scores]

        x = range(len(scores))
        plt.bar(x, baseline_scores, width=0.4, label='Baseline', align='center')
        plt.bar(x, custom_scores, width=0.4, label='Custom', align='edge')

        plt.xticks(x, scores)
        plt.ylabel('Scores')
        plt.title(f'{metric} Scores for Baseline and Custom Models')
        plt.legend()

        plt.savefig(f'generated_plots/{metric}_scores.png', bbox_inches='tight', transparent=True)
        plt.show()

if __name__ == "__main__":
    create_bar_graph()


# save to disk
def save_plot():
    #unimplemented
    pass



