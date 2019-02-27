import yaml

def load(settings_path):
    stream = open(settings_path, 'r')
    settings = yaml.load(stream)
    return settings