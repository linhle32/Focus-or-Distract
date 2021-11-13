#function to split videos into train, valid, and test
#videos are then further split into segments of *window* frames
#can set seed to replicate splits across different models
def train_valid_test_split(data, labels, test_rate, valid_rate, window, seed=123456):
    batch_data_size = len(data)
    ridx = np.arange(batch_data_size)
    np.random.seed(seed=seed)
    np.random.shuffle(ridx)
    test_ids = ridx[:int(batch_data_size*test_rate)]
    valid_ids = ridx[int(batch_data_size*test_rate):int(batch_data_size*test_rate)+int(batch_data_size*valid_rate)]
    train_ids = ridx[int(batch_data_size*test_rate)+int(batch_data_size*valid_rate):]

    def split_by_window(sequence, window, label):
        n = sequence.shape[0] // window
        psq = sequence.copy()
        psq[psq < 0] = -1
        fv = psq[:window*n].reshape(n, window, -1)
        l = [label] * n
        return fv, l

    X = []
    y = []
    for idx in [train_ids, valid_ids, test_ids]:
        tfd = []
        tl = []
        for i in idx:
            fv, l = split_by_window(data[i], window, labels[i])
            tfd.append(fv)
            tl.append(l)
        X.append(np.vstack(tfd))
        y.append(np.concatenate(tl))   

    X_train, X_valid, X_test = X
    y_train, y_valid, y_test = y

    return X_train, X_valid, X_test, y_train, y_valid, y_test