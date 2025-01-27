def float_to_category(component, val):
    r"""Map float value to component (category). """
    if val == 1:
        return len(component) - 1
    return int(val * len(component))


def float_to_num(component, val):
    r"""Map float value to integer. """
    parameters = [1] * len(component)
    for i in range(len(component)):
        parameters[i] = int(int(component[i]['min'] + (int(component[i]['max']) - int(component[i]['min'])) * val[i]))
    return parameters


def threshold(component, val):
    r"""Calculate whether feature is over a threshold. """
    data = [(i,c) for i, c in enumerate(component) if val[i] > 0.5]
    if data:
        return zip(*data)
    return [], ()

def calculate_dimension_of_the_problem(
        hyperparameters,
        evaluation_metrics,
        optimize_metric_weights=False):
    r"""Calculate the dimension of the problem. """

    metrics_factor = 1
    if optimize_metric_weights:
        metrics_factor = 2

    return (len(hyperparameters) + metrics_factor * len(evaluation_metrics) + 1)