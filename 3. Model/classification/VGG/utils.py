import matplotlib.pyplot as plt

def save_result(train_result, test_result):
    
    train_loss = train_result[0::2]
    train_acc = train_result[1::2]
    test_loss = test_result[0::2]
    test_acc = test_result[1::2]
    
    plt.figure(figsize=(10,10))
    
    plt.subplot(1,2,1)
    plt.plot(train_acc)
    plt.plot(test_acc)
    plt.legend(['Train_acc', 'Test_acc'])
    
    plt.subplot(1,2,2)
    plt.plot(train_loss)
    plt.plot(test_loss)
    plt.legend(['Train_loss', 'Test_loss'])
    
    print('Saving Result....')
    plt.savefig('result.jpg')
    
    return None
    
    