from data_converter.nuscenes_converter_seg import  create_nuscenes_infos

if __name__ == '__main__':
    # Training settings
    data_root = '/data/Dataset/nuScenes/'
    info_prefix = 'HDmaps-final'
    version='v1.0-trainval'
    # version == 'v1.0-test'
    # version == 'v1.0-mini'
    create_nuscenes_infos(data_root, info_prefix, version)

