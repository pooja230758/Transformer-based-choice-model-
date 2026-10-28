import os
import re

def replace_in_file(file_path, old, new):
    if not os.path.exists(file_path): return
    with open(file_path, 'r') as f:
        content = f.read()
    content = content.replace(old, new)
    with open(file_path, 'w') as f:
        f.write(content)

# Fix np.Inf
for f in ['choicemodels_real_data.py', 'choicemodels.py', 'Transformer.py', 'Transformer_nonpurchase.py']:
    replace_in_file(f, 'np.Inf', 'np.inf')

# Fix FEATURE_SIZES
if os.path.exists('choicemodels_real_data.py'):
    with open('choicemodels_real_data.py', 'r') as f:
        content = f.read()
    content = content.replace('FEATURE_SIZES = [6, 10, 2, 1, 1, 1, 1, 2, 1, 1, 2, 2]', 'FEATURE_SIZES = [11, 7, 97, 63, 2, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1]')
    content = content.replace('FIELD_SIZE = 12', 'FIELD_SIZE = 17')
    
    # MNL_loss fix
    old_mnl = """    def forward(self,x,y, temp_assortment_list):
        # batch_size,choice_size = x.shape
        batch_size = len(temp_assortment_list)
        loss = 0
        for i in range(batch_size):
            temp_sum = 0
            temp_y = 0
            # loss = loss + torch.log10((torch.exp(torch.sum(x[i]*y[i])))/(torch.sum(torch.exp(x[i]))))
            for j in temp_assortment_list[i]:
                temp_sum += torch.exp(x[j])
                if y[j] == 1:
                    temp_y = torch.exp(x[j])
            loss = loss + torch.log10(temp_y/(temp_sum))
        return -loss/batch_size"""
    new_mnl = """    def forward(self,x,y, temp_assortment_list):
        batch_size = len(temp_assortment_list)
        loss = 0
        for i in range(batch_size):
            temp_sum = 0
            temp_y = 0
            for j in temp_assortment_list[i]:
                temp_sum += torch.exp(x[j])
                if y[j] == 1:
                    temp_y = torch.exp(x[j])
            if temp_y == 0:
                loss = loss + torch.log10(1.0 / (1.0 + temp_sum))
            else:
                loss = loss + torch.log10(temp_y / (1.0 + temp_sum))
        return -loss/batch_size"""
    content = content.replace(old_mnl, new_mnl)

    # MMNL_loss fix
    old_mmnl = """    def forward(self,x,y,temp_assortment_list,z,alpha):
        batch_size = len(temp_assortment_list)
        model_num = alpha.shape[0]
        loss = 0
        for i in range(batch_size):
            g = 0
            for k in range(model_num):
                temp_sum = 0
                for j in temp_assortment_list[i]:
                    temp_sum += torch.exp(z[k,j])
                    if y[j] == 1:
                        temp_y = torch.exp(z[k,j])
                g = g + alpha[k]*temp_y/temp_sum
            g = float(g)
            loss = loss + torch.exp(torch.sum(x[temp_assortment_list[i]]*y[temp_assortment_list[i]]))/(torch.sum(torch.exp(x[temp_assortment_list[i]]))*g)
        return (-loss)/batch_size"""
    new_mmnl = """    def forward(self,x,y,temp_assortment_list,z,alpha):
        batch_size = len(temp_assortment_list)
        model_num = alpha.shape[0]
        loss = 0
        for i in range(batch_size):
            g = 0
            for k in range(model_num):
                temp_sum = 0
                temp_y = 0
                for j in temp_assortment_list[i]:
                    temp_sum += torch.exp(z[k,j])
                    if y[j] == 1:
                        temp_y = torch.exp(z[k,j])
                if temp_y == 0:
                    g = g + alpha[k] * 1.0 / (1.0 + temp_sum)
                else:
                    g = g + alpha[k] * temp_y / (1.0 + temp_sum)
            
            temp_y_x = 0
            temp_sum_x = 0
            for j in temp_assortment_list[i]:
                temp_sum_x += torch.exp(x[j])
                if y[j] == 1:
                    temp_y_x = torch.exp(x[j])
            if temp_y_x == 0:
                current_prob = 1.0 / (1.0 + temp_sum_x)
            else:
                current_prob = temp_y_x / (1.0 + temp_sum_x)
            
            loss = loss + current_prob / g
        return (-loss)/batch_size"""
    content = content.replace(old_mmnl, new_mmnl)
    with open('choicemodels_real_data.py', 'w') as f:
        f.write(content)

# Fix Assortment path
replace_in_file('training_function_real_data.py', 'Assortment/MNL_parameters_ztn.pt', 'trained_model_ztn/MNL_parameters_ztn.pt')
replace_in_file('training_function_real_data.py', 'Assortment\\\\MNL_parameters_ztn.pt', 'trained_model_ztn/MNL_parameters_ztn.pt')

