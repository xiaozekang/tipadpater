import os
import json
import random
from .utils import Datum, DatasetBase, read_json, write_json, build_data_loader


TEMPLATE = "a photo of {} with {} defect"

class MVTec_MC(DatasetBase):

    dataset_dir = 'mvtec'

    def __init__(self, root, num_shots, classname=None):
        
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir)

        with open('/data2/zekang/data/mvtec/category_tree_2.json', 'r') as f:
            category_tree = json.load(f)
        category_templates = category_tree[classname]
        classnames = []
        self.template = []
        def dfs(tree):
            for cat in tree.keys():
                if 'next' in tree[cat].keys():
                    dfs(tree[cat]['next'])
                else:
                    classnames.append(cat)
                    self.template.append(tree[cat]['text'])
        dfs(category_templates)
        # print(self.template, classnames)
        # self.template = [TEMPLATE.format(classname, '{}')]

        # classnames = []
        atype2lab = {}
        # classnames = os.listdir(os.path.join(self.dataset_dir, classname, 'test'))
        for idx, atype in enumerate(classnames):
            if atype == 'good':
                atype2lab['no'] = idx
            else:
                atype2lab[atype] = idx

        # train = self.read_data(cname2lab, 'images_variant_train.txt')
        # val = self.read_data(cname2lab, 'images_variant_val.txt')
        # test = self.read_data(cname2lab, 'images_variant_test.txt')
        train, val, test = self.read_data(atype2lab, 'split_2.json', classname)
        
        train = self.generate_fewshot_dataset(train, num_shots=num_shots)
        
        super().__init__(train_x=train, val=val, test=test)
    
    def read_data(self, cname2lab, split_file, classname):
        filepath = os.path.join(self.dataset_dir, split_file)
        train_set = []
        val_set = []
        test_set = []
        
        with open(filepath, 'r') as f:
            split = json.load(f)
        anno_data = split[classname]
        
        for anomaly_type in anno_data['train'].keys():
            # print('split train:')
            # print(anno_data['train'][anomaly_type])
            for imname in anno_data['train'][anomaly_type]:
                impath = os.path.join(self.image_dir, imname)
                label = cname2lab[anomaly_type] if anomaly_type != 'good' else cname2lab['no']
                item = Datum(
                    impath=impath,
                    label=label,
                    classname=anomaly_type if anomaly_type != 'good' else 'no'
                )
                train_set.append(item)
        
        for anomaly_type in anno_data['test'].keys():
            val_img = random.sample(anno_data['test'][anomaly_type], k=len(anno_data['test'][anomaly_type])//2)
            test_img = anno_data['test'][anomaly_type]
            # print('split val and test:')
            # print(val_img, test_img)
            for imname in val_img:
                impath = os.path.join(self.image_dir, imname)
                label = cname2lab[anomaly_type] if anomaly_type != 'good' else cname2lab['no']
                item = Datum(
                    impath=impath,
                    label=label,
                    classname=anomaly_type if anomaly_type != 'good' else 'no'
                )
                val_set.append(item)
            for imname in test_img:
                impath = os.path.join(self.image_dir, imname)
                label = cname2lab[anomaly_type] if anomaly_type != 'good' else cname2lab['no']
                item = Datum(
                    impath=impath,
                    label=label,
                    classname=anomaly_type if anomaly_type != 'good' else 'no'
                )
                test_set.append(item)

        return train_set, val_set, test_set

if __name__ == "__main__":
    dataset = MVTec_MC(root='/data2/zekang/data', num_shots=2, classname='cable')
    print(dataset.classnames)