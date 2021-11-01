import os

import matplotlib.pyplot as plt

def save_result(train_result, test_result, result_path):
    
    train_loss = train_result[0::2]
    train_acc = train_result[1::2]
    test_loss = test_result[0::2]
    test_acc = test_result[1::2]
    
    plt.figure(figsize=(12,7))
    plt.subplot(1,2,1)
    plt.plot(train_acc)
    plt.plot(test_acc)
    plt.legend(['Train_acc', 'Test_acc'])
    
    plt.subplot(1,2,2)
    plt.plot(train_loss)
    plt.plot(test_loss)
    plt.legend(['Train_loss', 'Test_loss'])
    
    plt.savefig(f'{result_path}/result.jpg')
    
    return None
    
    
def make_folder(base_path, folder):
    os.makedirs(f'./{base_path}', exist_ok=True)
    try:
        os.mkdir(f'./{base_path}/{folder}')
        new_folder = f'./{base_path}/{folder}'
        
    except:
        exist_folders = os.listdir(f'./{base_path}')
        exist_folders.sort(key=lambda x: int(x[5:]))
        
        last_num = exist_folders[-1][5:]
        new_num = int(last_num) + 1
        new_folder = f'./{base_path}/train' + str(new_num)
        
        os.mkdir(new_folder)
        
    return new_folder
    