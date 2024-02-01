from pytorch_model_summary import summary

def model_summary(model, fake_input, device):
    print(summary(model, fake_input.to(device), max_depth=True, show_parent_layers=True))
    print(model)
    return