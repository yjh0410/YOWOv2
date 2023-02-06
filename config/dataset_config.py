# Dataset configuration


dataset_config = {
    'ucf24': {
        # dataset
        'gt_folder': './evaluator/groundtruths_ucf_jhmdb/groundtruths_ucf/',
        # input size
        'train_size': 224,
        'test_size': 224,
        # transform
        'jitter': 0.2,
        'hue': 0.1,
        'saturation': 1.5,
        'exposure': 1.5,
        'sampling_rate': 1,
        # cls label
        'multi_hot': False,  # one hot
        # optimizer
        'optimizer': 'adamw',
        'momentum': 0.9,
        'weight_decay': 5e-4,
        # warmup strategy
        'warmup': 'linear',
        'warmup_factor': 0.00066667,
        'wp_iter': 500,
        # class names
        'valid_num_classes': 24,
        'label_map': (
                    'Basketball',     'BasketballDunk',    'Biking',            'CliffDiving',
                    'CricketBowling', 'Diving',            'Fencing',           'FloorGymnastics', 
                    'GolfSwing',      'HorseRiding',       'IceDancing',        'LongJump',
                    'PoleVault',      'RopeClimbing',      'SalsaSpin',         'SkateBoarding',
                    'Skiing',         'Skijet',            'SoccerJuggling',    'Surfing',
                    'TennisSwing',    'TrampolineJumping', 'VolleyballSpiking', 'WalkingWithDog'
                ),
    },
    
    'ava_v2.2':{
        # dataset
        'frames_dir': 'frames/',
        'frame_list': 'frame_lists/',
        'annotation_dir': 'annotations/',
        'train_gt_box_list': 'ava_v2.2/ava_train_v2.2.csv',
        'val_gt_box_list': 'ava_v2.2/ava_val_v2.2.csv',
        'train_exclusion_file': 'ava_v2.2/ava_train_excluded_timestamps_v2.2.csv',
        'val_exclusion_file': 'ava_v2.2/ava_val_excluded_timestamps_v2.2.csv',
        'labelmap_file': 'ava_v2.2/ava_action_list_v2.2_for_activitynet_2019.pbtxt', # 'ava_v2.2/ava_action_list_v2.2.pbtxt',
        'class_ratio_file': 'config/ava_categories_ratio.json',
        'backup_dir': 'results/',
        # input size
        'train_size': 224,
        'test_size': 224,
        # transform
        'jitter': 0.2,
        'hue': 0.1,
        'saturation': 1.5,
        'exposure': 1.5,
        'sampling_rate': 1,
        # cls label
        'multi_hot': True,  # multi hot
        # train config
        'optimizer': 'adamw',
        'momentum': 0.9,
        'weight_decay': 5e-4,
        # warmup strategy
        'warmup': 'linear',
        'warmup_factor': 0.00066667,
        'wp_iter': 500,
        # class names
        'valid_num_classes': 80,
        'label_map': (
                    'bend/bow(at the waist)', 'crawl', 'crouch/kneel', 'dance', 'fall down',  # 1-5
                    'get up', 'jump/leap', 'lie/sleep', 'martial art', 'run/jog',             # 6-10
                    'sit', 'stand', 'swim', 'walk', 'answer phone',                           # 11-15
                    'brush teeth', 'carry/hold (an object)', 'catch (an object)', 'chop', 'climb (e.g. a mountain)',  # 16-20
                    'clink glass', 'close (e.g., a door, a box)', 'cook', 'cut', 'dig',                               # 21-25
                    'dress/put on clothing', 'drink', 'drive (e.g., a car, a truck)', 'eat', 'enter',                 # 26-30
                    'exit', 'extract', 'fishing', 'hit (an object)', 'kick (an object)',                              # 31-35
                    'lift/pick up', 'listen (e.g., to music)', 'open (e.g., a window, a car door)', 'paint', 'play board game',  # 36-40
                    'play musical instrument', 'play with pets', 'point to (an object)', 'press','pull (an object)',             # 41-45
                    'push (an object)', 'put down', 'read', 'ride (e.g., a bike, a car, a horse)', 'row boat',                   # 46-50
                    'sail boat', 'shoot', 'shovel', 'smoke', 'stir',                                                             # 51-55
                    'take a photo', 'text on/look at a cellphone', 'throw', 'touch (an object)', 'turn (e.g., a screwdriver)',   # 56-60
                    'watch (e.g., TV)', 'work on a computer', 'write', 'fight/hit (a person)', 'give/serve (an object) to (a person)',  # 61-65
                    'grab (a person)', 'hand clap', 'hand shake', 'hand wave', 'hug (a person)',                                        # 66-70
                    'kick (a person)', 'kiss (a person)', 'lift (a person)', 'listen to (a person)', 'play with kids',                  # 71-75
                    'push (another person)', 'sing to (e.g., self, a person, a group)', 'take (an object) from (a person)',             # 76-78
                    'talk to (e.g., self, a person, a group)', 'watch (a person)'                                                       # 79-80
                ),
    }
}
