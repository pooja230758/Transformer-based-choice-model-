import pandas as pd
import numpy as np
import build_new_data as bnd
import true_model

order_num = 10
choice_size = 30
c_rate = 5
next_train_data = bnd.generate_data(order_num, int(choice_size+c_rate))
pro = true_model.true_model(next_train_data, order_num)
print("Success, shape:", pro.shape)
