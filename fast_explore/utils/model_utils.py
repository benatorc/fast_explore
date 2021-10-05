import torch

def save_model(save_path, model, model_class, args, verbose=False):
    model_dict = {
        'model': model.state_dict(),
        'model_class': model_class,
        'args': args, }
    torch.save(model_dict, save_path)

    if verbose:
        print('Model saved at: %s' % save_path)


def load_model(save_path):
    model_dict = torch.load(save_path)
    model = model_dict['model_class'](model_dict['args']).cuda()

    model.load_state_dict(model_dict['model'])
    return model
