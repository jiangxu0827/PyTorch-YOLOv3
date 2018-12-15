

def parse_model_config(path):
    """Parses the yolo-v3 layer configuration file and returns module definitions"""
    file = open(path, 'r')
    lines = file.read().split('\n')
    lines = [x for x in lines if x and not x.startswith('#')]
    lines = [x.rstrip().lstrip() for x in lines] # get rid of fringe whitespaces
    module_defs = []
    for line in lines:
        if line.startswith('['): # This marks the start of a new block
            module_defs.append({})
            module_defs[-1]['type'] = line[1:-1].rstrip()
            if module_defs[-1]['type'] == 'convolutional':
                module_defs[-1]['batch_normalize'] = 0
        else:
            key, value = line.split("=")
            value = value.strip()
            module_defs[-1][key.rstrip()] = value.strip()

    return module_defs

def parse_data_config(path):
    """Parses the data configuration file"""
    options = dict()
    options['gpus'] = '0,1,2,3'
    options['num_workers'] = '10'
    with open(path, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.strip()
        if line == '' or line.startswith('#'):
            continue
        key, value = line.split('=')
        options[key.strip()] = value.strip()
    return options

def model_filter(model):
    for i,m in enumerate(model.modules()):
        for p in m.parameters():
            p.requires_grad=False
    #for i,m in enumerate(model.modules()):
        if i==361:
            for p in m.parameters():
                p.requires_grad=True
        elif i==357:
            for p in m.parameters():
                p.requires_grad=True
        elif i==353:
            for p in m.parameters():
                p.requires_grad=True
        elif i==349:
            for p in m.parameters():
                p.requires_grad=True
        elif i==345:
            for p in m.parameters():
                p.requires_grad=True
        elif i==341:
            for p in m.parameters():
                p.requires_grad=True
        elif i==337:
            for p in m.parameters():
                p.requires_grad=True 
        elif i==333:
            for p in m.parameters():
                p.requires_grad=True
        elif i==329:
            for p in m.parameters():
                p.requires_grad=True
                #for 3rd yolo layer
        elif i==321:
            for p in m.parameters():
                p.requires_grad=True
        elif i==317:
            for p in m.parameters():
                p.requires_grad=True
        elif i==313:
            for p in m.parameters():
                p.requires_grad=True
        elif i==309:
            for p in m.parameters():
                p.requires_grad=True
        elif i==305:
            for p in m.parameters():
                p.requires_grad=True
        elif i==301:
            for p in m.parameters():
                p.requires_grad=True 
        elif i==297:
            for p in m.parameters():
                p.requires_grad=True 
        elif i==293:
            for p in m.parameters():
                p.requires_grad=True
        elif i==289:
            for p in m.parameters():
                p.requires_grad=True
                #for 2nd yolo layer
        elif i==281:
            for p in m.parameters():
                p.requires_grad=True
        elif i==277:
            for p in m.parameters():
                p.requires_grad=True
        elif i==273:
            for p in m.parameters():
                p.requires_grad=True 
        elif i==269:
            for p in m.parameters():
                p.requires_grad=True
        elif i==265:
            for p in m.parameters():
                p.requires_grad=True
        elif i==261:
            for p in m.parameters():
                p.requires_grad=True 
                 #for 1st yolo layer
    return model