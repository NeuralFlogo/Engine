def initialize(model, init_func, *params, **kwargs):
    for p in model.parameters():
        init_func(p, *params, **kwargs)
