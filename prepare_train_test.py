#!env python3

# create csv label files based on dir name
# sub1/1.png sub1/2.png sub2/1.png will produce following content
# file,label
# sub1/1.png,1
# sub1/2.png,1
# sub2/1.png,1

import csv
import os
import sys


# todo: use sklearn cross validation
def split_train_test(fn_dir, train_max=10000):
    train = []
    test = []
    id = 0
    for subdirs, dirs, files in os.walk(fn_dir):
        for subdir in dirs:
            subject_path = os.path.join(fn_dir, subdir)
            cnt = 1
            for fname in os.listdir(subject_path):
                name, ext = os.path.splitext(fname)
                if ext.lower() not in ['.png', 'jpg', 'jpeg', 'gif', '.pgm']:
                    print('skipping ', fname, ' as wrong file type')
                    continue
                path = os.path.join(subject_path, fname)
                label = id

                if cnt <= train_max:
                    train.append({
                        'image': path,
                        'label': int(label),
                        'name': subdir
                    })
                else:
                    test.append({
                        'image': path,
                        'label': int(label),
                        'name': subdir
                    })
                cnt += 1
            id += 1

    return train, test


def write_dicts_to_csv(data, path):
    if data:
        keys = data[0].keys()
        with open(path, 'w') as output_file:
            dict_writer = csv.DictWriter(output_file, keys)
            dict_writer.writeheader()
            dict_writer.writerows(data)
    else:
        print('WARN: data is empty')


if __name__ == '__main__':
    try:
        fn_name = sys.argv[1]
        train_path = sys.argv[2]
        test_path = sys.argv[3]
    except IndexError:
        print('please provide a parent dir name & train output path & test output path')
        sys.exit(1)

    train, test = split_train_test(fn_name, train_max=8)
    write_dicts_to_csv(train, train_path)
    write_dicts_to_csv(test, test_path)
    print('DONE')
