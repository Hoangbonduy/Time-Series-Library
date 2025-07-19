from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_M4, PSMSegLoader, \
    MSLSegLoader, SMAPSegLoader, SMDSegLoader, SWATSegLoader, UEAloader
from data_provider.uea import collate_fn
from torch.utils.data import DataLoader

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
    'm4': Dataset_M4,
    'PSM': PSMSegLoader,
    'MSL': MSLSegLoader,
    'SMAP': SMAPSegLoader,
    'SMD': SMDSegLoader,
    'SWAT': SWATSegLoader,
    'UEA': UEAloader
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    shuffle_flag = False if (flag == 'test' or flag == 'TEST') else True
    drop_last = False
    batch_size = args.batch_size
    freq = args.freq

    if args.task_name == 'anomaly_detection':
        drop_last = False
        
        # --- BẮT ĐẦU SỬA ĐỔI ---
        # Kiểm tra xem có phải chúng ta đang dùng một trong các SegLoader không
        if args.data in ['PSM', 'MSL', 'SMAP', 'SMD', 'SWAT']:
            # Nếu đúng, truyền win_size
            data_set = Data(
                args=args,
                root_path=args.root_path,
                win_size=args.seq_len,
                flag=flag,
            )
        else:
            # --- BẮT ĐẦU SỬA ĐỔI ---
            
            # Xác định tên file dựa trên flag
            if flag == 'train':
                data_path = 'train.csv'
            elif flag == 'test':
                data_path = 'test.csv'
            else: # flag == 'val'
                # Trong trường hợp của bạn, không có file val riêng,
                # có thể dùng file test hoặc báo lỗi tùy logic
                data_path = 'test.csv' 

            data_set = Data(
                args=args,
                root_path=args.root_path,
                data_path=data_path, # <--- TRUYỀN TÊN FILE ĐÚNG VÀO ĐÂY
                flag=flag,
                size=[args.seq_len, args.label_len, args.pred_len],
                features=args.features,
                target=args.target,
                timeenc=timeenc,
                freq=freq
            )
            # --- KẾT THÚC SỬA ĐỔI ---

        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader
    else:
        if args.data == 'm4':
            drop_last = False
        data_set = Data(
            args = args,
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            seasonal_patterns=args.seasonal_patterns
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader
