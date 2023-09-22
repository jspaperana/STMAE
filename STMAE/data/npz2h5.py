import numpy as np 
import pandas as pd

if __name__ == "__main__":
    num_list = ['3', '4', '7', '8']
    start_list = ['2018-09-01', '2018-01-01', '2017-05-01', '2016-07-01']
    end_list = ['2018-12-01', '2018-03-01', '2017-08-07', '2016-09-01']

    for idx, num in enumerate(num_list):
        print(idx)
        data = np.load('data/PEMS0'+num+'/PEMS0'+num+'.npz')['data']
        print(data.shape)

        df = pd.DataFrame(data[:,:,0])
        print(df.shape)

        date = pd.date_range(start=start_list[idx], end=end_list[idx], freq='5T')[:-1]
        print(date.shape)
        df = df.set_index(date)

        df.to_hdf('pems-0'+num+'.h5', key='t', mode='w')