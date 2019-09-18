from frontend import KneeCAM
from dict_models import DefModel
from fusion import fusion


def main():
    num_case = 0
    CAM = KneeCAM(out_class=2)
    for num_loc in range(0, 23):
        CAM.Manager.options['slice_range'][0] = list(range(num_loc, num_loc + 1, 1))
        CAM.Manager.options['slice_range'][1] = list(range(num_loc, num_loc + 1, 1))

        """ Initialize """
        text_name = 'out_' + CAM.Manager.options['network_choice'] + '_' + CAM.Manager.options['fusion_method'] + '_' \
                    + str(num_loc) + '.txt'

        model_ini = DefModel(Manager=CAM.Manager, zlen=len(CAM.Manager.options['slice_range'][0]), out_class=2)

        CAM.Manager.init_model(model_ini)
        CAM.prep_training(num_case, num_loc, text_name, training='training', cont=False)

    fusion(CAM.Manager, num_case, 512)


if __name__ == "__main__":
    main()
