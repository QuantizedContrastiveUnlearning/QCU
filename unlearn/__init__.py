from .QCU import QCU


def get_unlearn_method(name):
    """method usage:

    function(data_loaders, model, criterion, args)"""
    if name == "QCU":
        return QCU
    else:
        raise NotImplementedError(f"Unlearn method {name} not implemented!")
