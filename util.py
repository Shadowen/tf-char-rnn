class lazyproperty(object):
    '''
    meant to be used for lazy evaluation of an object attribute.
    property should represent non-mutable datasets, as it replaces itself.
    '''

    def __init__(self, fget):
        self.fget = fget
        self.func_name = fget.__name__

    def __get__(self, obj, cls):
        if obj is None:
            return None
        value = self.fget(obj)
        setattr(obj, self.func_name, value)
        return value


def get_unique_deterministic(lst):
    unique_list = []
    for item in lst:
        if item not in unique_list:
            unique_list.append(item)
    return unique_list
