import matplotlib.pyplot as plt

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


