import numpy as np
from keras.models import load_model

db = "imdb"
data_path = f"data/{db}_arrays_cropped_faces/"
model_number = '45m15ev3_it'

array_num = 0
print(f'loading {data_path}images_{db}_{array_num}.npy')
X = np.load(f'{data_path}images_{db}_{array_num}.npy')
print(f'loading {data_path}gender_{db}_{array_num}.npy')
y = np.load(f'{data_path}gender_{db}_{array_num}.npy')

eval_loss = []
eval_perc = []

for i in range(46):
    model = load_model(f'models/imdb_model_gender_{model_number}{i}.h5')
    evaluated = model.evaluate(X, y, batch_size=50)
    print(evaluated)
    eval_loss.append(evaluated[0])
    eval_perc.append(evaluated[1])

print(eval_loss)
eval_loss.plot(title="loss")
print(eval_perc)
eval_perc.plot(title="perc")

import matplotlib.pyplot as plt
plt.plot(eval_loss)
plt.plot(eval_perc)

eval_loss1 = np.array(eval_loss)
eval_perc1 = np.array(eval_perc)
np.save('loss', np.array(eval_loss))
np.save('perc', np.array(eval_perc))
    
#y_hat = model.predict(X)
#ans = 0
#for i in range(len(y)):
#    if y_hat[i,0] > y_hat[i,1]:
#        detected_gender = [1,0]
#    else:
#        detected_gender = [0,1]
#    if list(y[i]) == detected_gender:
#        ans+=1
#        print (f"{i} detected correct")
#    else:
#        print (f"{i} detected INcorrect")
#print (ans/len(y)*100)
#        

#
#for array_num in range(0, arrays_number+1):
#    print(f'loading {data_path}images_{db}_{array_num}.npy')
#    X = np.load(f'{data_path}images_{db}_{array_num}.npy')
#    print(f'loading {data_path}gender_{db}_{array_num}.npy')
#    y = np.load(f'{data_path}gender_{db}_{array_num}.npy')
#    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#    model.fit(X_train, y_train, 
#              epochs=epochs_number,
#              batch_size=50,
#              validation_data=(X_test, y_test))
#    print (f'saving model models/{db}_model_gender_{model_number}_it{array_num}.h5')
#    model.save(f'models/{db}_model_gender_{model_number}_it{array_num}.h5')
#
#
#
#print ("females")
#for i in range(1,11):
#    detected_gender, woman_score, man_score = gender(i, 'f')
#    print (f"image {i} is a {detected_gender}, score is {woman_score} / {man_score}")
#    
#print ("males")
#for i in range(1,11):
#    detected_gender, woman_score, man_score = gender(i, 'm')
#    print (f"image {i} is a {detected_gender}, score is {woman_score} / {man_score}")