with open('training_function.py', 'r') as f:
    content = f.read()

content = content.replace('model_list = [torch.load("Assortment\\\\MNL_parameters_"+str(c_rate, weights_only=False)+".pt", weights_only=False)]', 'model_list = [torch.load("Assortment\\\\MNL_parameters_"+str(c_rate)+".pt", weights_only=False)]')

with open('training_function.py', 'w') as f:
    f.write(content)
