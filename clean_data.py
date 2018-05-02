import numpy as np
import pandas as pd
# import scipy

from sklearn.preprocessing import OneHotEncoder
pd.options.mode.chained_assignment = None

if __name__ == '__main__':
    path_train = './mnt/ssd/kaggle-talkingdata2/competition_files/train.csv'
    train_cols = [
        'ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed'
    ]
    dtypes = {
        'ip': 'uint32',
        'app': 'uint16',
        'device': 'uint16',
        'os': 'uint16',
        'channel': 'uint16',
        'is_attributed': 'uint8',
        'click_id': 'uint32',
        'is_attributed': 'bool'
    }
    usecols = ['app', 'os', 'channel', 'click_time', 'is_attributed']

    nrows = 50000000

    print('Loading the data...')
    df = pd.read_csv(
        path_train, dtype=dtypes, header=0, usecols=usecols, nrows=nrows)
    print('End loading data...')

    print('Grabbing labels')
    # Get labels and save them separately
    y = df.pop('is_attributed')
    print('Grabbed labels')

    print('Processing app column')
    # Process 'app' column
    app = df['app'].value_counts()
    app_90 = ((app / app.sum()).cumsum() < .9).sum()
    app_90 = app.index[:app_90]
    df['app'][~df['app'].isin(app_90)] = 9999
    print('End processing app column')

    print('Processing os column')
    # Process 'os' column
    os = df['os'].value_counts()
    os_90 = ((os / os.sum()).cumsum() < .9).sum()
    os_90 = os.index[:os_90]
    df['os'][~df['os'].isin(os_90)] = 9999
    print('End processing os column')

    print('Processing channel column')
    # Process 'channel' column
    channel = df['channel'].value_counts()
    channel_90 = ((channel / channel.sum()).cumsum() < .9).sum()
    channel_90 = channel.index[:channel_90]
    df['channel'][~df['channel'].isin(channel_90)] = 9999
    print('End processing channel column')

    print('Processing time column')
    time = df.pop('click_time')
    time = time.str.slice(11, 13).values.astype(np.uint8)
    time = time.reshape((-1, 1))
    print('End processing channel column')

    X = df[['app', 'os', 'channel']].values

    X = np.concatenate([X, time], axis=1)

    print('One hot encoding all columns')
    enc = OneHotEncoder(dtype=np.bool, sparse=False)
    X = enc.fit_transform(X)
    print('End one hot encoding')

    print('Saving cleaned data')
    np.savez_compressed('clean_data', X=X, y=y)
    print('Saved')

    print('Make sure that the following number is 119')
    print(X.shape[1])