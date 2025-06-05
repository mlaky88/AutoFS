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

def float_to_params(component, val):
    #print(component, val)
    return component['min'] + (component['max'] - component['min']) * val
    


def threshold(component, val):
    r"""Calculate whether feature is over a threshold. """
    data = [(i,c) for i, c in enumerate(component) if val[i] > 0.5]
    if data:
        return zip(*data)
    return [], ()

def threshold_filter_methods(filter_methods, val):
    r"""Calculate whether feature is over a threshold. """

    i = 0
    data = []
    idxs = []
    #print(filter_methods)
    for filt in filter_methods:
        num_keys = len(filt.keys())
        x = val[i:i + len(filt.keys())]
        if x[0] > 0.5: # if the first value is over the threshold
            f = {'name': filt['name']}
            for j in range(1, len(filt.keys())):
                key = list(filt.keys())[j]
                f[key] = filt[key]['min'] + x[j] * (filt[key]['max'] - filt[key]['min'])
            data.append(f)
            idxs.append(j-1)
        i += num_keys

    if (data):
        return idxs, data
    return [], []
    
def calculate_dimension_of_the_problem(
        hyperparameters,
        filter_methods,
        pipeline_evaluation_algorithm,
        optimize_metric_weights=False):
    r"""Calculate the dimension of the problem. """

    metrics_factor = 1
    if optimize_metric_weights:
        metrics_factor = 2

    num_hyper_parameters = len(hyperparameters)
    num_filter_methods = len(filter_methods)
    num_metrics = 0#len(pipeline_evaluation_algorithm) - 1 # TODO currently not optimizing the classifier (KNN)
    #print("Metrics:", num_metrics)
    #print(pipeline_evaluation_algorithm)
    x = 0
    for filt_method in filter_methods:
        x += len(filt_method.keys()) - 1

    return 2 + num_hyper_parameters + num_metrics + x + metrics_factor * num_filter_methods