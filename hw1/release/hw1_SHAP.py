import sys, csv, pickle
import pandas as pd
import numpy as np

test_file = sys.argv[1]
output_file = sys.argv[2]

pickle_file = './reg.pickle'
mean_x = np.load('./mean_x.npy')
std_x = np.load('./std_x.npy')
importance = list(np.load('./importance.npy'))

with open(pickle_file, 'rb') as f:
     reg = pickle.load(f)

testdata = pd.read_csv(
  test_file,
  header = None,
  encoding = 'big5'
)
test_data = testdata.iloc[:, 2:]
test_data[test_data == 'NR'] = 0
test_data = test_data.to_numpy()

test_x = np.empty([240, 18*9], dtype = float)
for i in range(240):
    test_x[i, :] = test_data[18 * i: 18* (i + 1), :].reshape(1, -1)
for i in range(len(test_x)):
    for j in range(len(test_x[0])):
        if std_x[j] != 0:
            test_x[i][j] = (test_x[i][j] - mean_x[j]) / std_x[j]
test_x = np.concatenate((np.ones([240, 1]), test_x), axis = 1).astype(float)

X_test_best = pd.DataFrame(test_x)
X_test_selected = X_test_best.loc[:,importance]

ans_y = reg.predict(X_test_selected)

with open(output_file, mode='w', newline='') as submit_file:
    csv_writer = csv.writer(submit_file)
    header = ['id', 'value']
    print(header)
    csv_writer.writerow(header)
    for i in range(240):
        row = ['id_' + str(i), ans_y[i][0]]
        csv_writer.writerow(row)
        print(row)