import torch, os
from util import get_args, KneeManager, prep_training, KneeData
from util import get_label_KL_uni as get_label
from dict_models import DefModel
from frontend import KneeCAM


def main():

    CAM = KneeCAM()

    for case_num in range(2, 3):
        for slice_loc in range(0, 23):
            CAM.Options['slice_range'] = [[slice_loc], [slice_loc]]
            slice_name = 'SAG_' + str(slice_loc)
            """ Training """
            CAM.train(case_num, slice_name, dataset, testing=False, cont=False)



if __name__ == "__main__":
    main()




