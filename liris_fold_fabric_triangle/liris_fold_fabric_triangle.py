from typing import Iterator, Tuple, Any

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import os
from PIL import Image
import datetime

from liris_fold_fabric_triangle.utils import load_trajectory, crawler
from liris_fold_fabric_triangle.tfds_utils import MultiThreadedDatasetBuilder


# We assume a fixed language instruction here -- if your dataset has various instructions, please modify
LANGUAGE_INSTRUCTION = 'Fold the fabric into a triangle'
                        
# Modify to point to directory with raw DROID MP4 data
DATA_PATH = "/home/panda/liris_droid/data/success"

# (180, 320) is the default resolution, modify if different resolution is desired
IMAGE_RES = (360, 640)

# Set the dates from which you want to compose the dataset
INITIAL_DATE = datetime.datetime(2025, 3, 18, 9, 55, 00)
FINAL_DATE = datetime.datetime(2025, 3, 18, 11, 00, 00)

# Filter timesteps in which no command is given to the robot
FILTER_NO_OPS = True


def _generate_examples(paths) -> Iterator[Tuple[str, Any]]:

    def _resize_and_encode(image, size):
        image = Image.fromarray(image)
        return np.array(image.resize(size, resample=Image.BICUBIC))

    def _parse_example(episode_path):
        h5_filepath = os.path.join(episode_path, 'trajectory.h5')
        recording_folderpath = os.path.join(episode_path, 'recordings', 'MP4')

        try:
            data = load_trajectory(h5_filepath, recording_folderpath=recording_folderpath)
        except:
           print(f"Skipping trajectory because data couldn't be loaded for {episode_path}.")
           return None

        # get a random language instruction
        lang = LANGUAGE_INSTRUCTION

        try:
            assert all(t.keys() == data[0].keys() for t in data)
            for t in range(len(data)):
                for key in data[0]['observation']['image'].keys():
                    data[t]['observation']['image'][key] = _resize_and_encode(
                        data[t]['observation']['image'][key], (IMAGE_RES[1], IMAGE_RES[0])
                    )

            # assemble episode --> here we're assuming demos so we set reward to 1 at the end
            episode = []

            for step in data:
                if FILTER_NO_OPS and (step['action']['cartesian_velocity'] == 0).all():
                    continue

                obs = step['observation']
                action = step['action']
                camera_type_dict = obs['camera_type']
                wrist_ids = [k for k, v in camera_type_dict.items() if v == 0]
                exterior_ids = [k for k, v in camera_type_dict.items() if v != 0]

                episode.append({
                    'observation': {
                        'exterior_image_1_left': obs['image'][f'{exterior_ids[0]}_left'][..., ::-1],
                        'wrist_image_left': obs['image'][f'{wrist_ids[0]}_left'][..., ::-1],
                        'cartesian_position': obs['robot_state']['cartesian_position'],
                        'joint_position': obs['robot_state']['joint_positions'],
                        'gripper_position': np.array([obs['robot_state']['gripper_position']]),
                    },
                    'action_dict': {
                        'cartesian_position': action['cartesian_position'],
                        'cartesian_velocity': action['cartesian_velocity'],
                        'gripper_position': np.array([action['gripper_position']]),
                        'gripper_velocity': np.array([action['gripper_velocity']]),
                        'joint_position': action['joint_position'],
                        'joint_velocity': action['joint_velocity'],
                    },
                    'action': np.concatenate((action['cartesian_position'], [action['gripper_position']])),
                    'discount': 1.0,
                    'reward': 0.0,
                    'is_first': False,
                    'is_last': False,
                    'is_terminal': False,
                    'language_instruction': lang,
                })

            episode[0]["is_first"] = True
            episode[-1]["is_last"] = True
            episode[-1]["is_terminal"] = True
            episode[-1]["reward"] = 1.0

        except:
           print(f"Skipping trajectory because there was an error in data processing for {episode_path}.")
           return None

        # create output data sample
        sample = {
            'steps': episode,
            'episode_metadata': {
                'file_path': h5_filepath,
                'recording_folderpath': recording_folderpath
            }
        }
        # if you want to skip an example for whatever reason, simply return None
        return episode_path, sample

    # for smallish datasets, use single-thread parsing
    for sample in paths:
       yield _parse_example(sample)


class LirisFoldFabricTriangle(MultiThreadedDatasetBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
    }

    N_WORKERS = 4                  # number of parallel workers for data conversion
    MAX_PATHS_IN_MEMORY = 20       # number of paths converted & stored in memory before writing to disk
                                    # -> the higher the faster / more parallel conversion, adjust based on avilable RAM
                                    # note that one path may yield multiple episodes and adjust accordingly
    PARSE_FCN = _generate_examples  # handle to parse function from file paths to RLDS episodes

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
            'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'exterior_image_1_left': tfds.features.Image(
                            shape=(*IMAGE_RES, 3),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='Exterior camera 1 left viewpoint',
                        ),
                        'wrist_image_left': tfds.features.Image(
                            shape=(*IMAGE_RES, 3),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='Wrist camera RGB left viewpoint',
                        ),
                        'cartesian_position': tfds.features.Tensor(
                            shape=(6,),
                            dtype=np.float64,
                            doc='Robot Cartesian state',
                        ),
                        'gripper_position': tfds.features.Tensor(
                            shape=(1,),
                            dtype=np.float64,
                            doc='Gripper position statae',
                        ),
                        'joint_position': tfds.features.Tensor(
                            shape=(7,),
                            dtype=np.float64,
                            doc='Joint position state'
                        )
                    }),
                    'action_dict': tfds.features.FeaturesDict({
                        'cartesian_position': tfds.features.Tensor(
                            shape=(6,),
                            dtype=np.float64,
                            doc='Commanded Cartesian position'
                        ),
                        'cartesian_velocity': tfds.features.Tensor(
                            shape=(6,),
                            dtype=np.float64,
                            doc='Commanded Cartesian velocity'
                        ),
                        'gripper_position': tfds.features.Tensor(
                            shape=(1,),
                            dtype=np.float64,
                            doc='Commanded gripper position'
                        ),
                        'gripper_velocity': tfds.features.Tensor(
                            shape=(1,),
                            dtype=np.float64,
                            doc='Commanded gripper velocity'
                        ),
                        'joint_position': tfds.features.Tensor(
                            shape=(7,),
                            dtype=np.float64,
                            doc='Commanded joint position'
                        ),
                        'joint_velocity': tfds.features.Tensor(
                            shape=(7,),
                            dtype=np.float64,
                            doc='Commanded joint velocity'
                        )
                    }),
                    'action': tfds.features.Tensor(
                        shape=(7,),
                        dtype=np.float64,
                        doc='Robot action, consists of [6x joint positions, \
                            1x gripper position].',
                    ),
                    'discount': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Discount if provided, default to 1.'
                    ),
                    'reward': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Reward if provided, 1 on final step for demos.'
                    ),
                    'is_first': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on first step of the episode.'
                    ),
                    'is_last': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode.'
                    ),
                    'is_terminal': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode if it is a terminal step, True for demos.'
                    ),
                    'language_instruction': tfds.features.Text(
                        doc='Language Instruction.'
                    ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                    'recording_folderpath': tfds.features.Text(
                        doc='Path to the folder of recordings.'
                    )
                }),
            }))

    def _split_paths(self):
        """Define data splits."""
        # create list of all examples -- by default we put all examples in 'train' split
        # add more elements to the dict below if you have more splits in your data
        print("Crawling all episode paths...")
        episode_paths = crawler(DATA_PATH)
        episode_paths = [p for p in episode_paths if os.path.exists(p + '/trajectory.h5') and \
                         os.path.exists(p + '/recordings/MP4')]
        
        if INITIAL_DATE:
            episode_paths = [e for e in episode_paths if datetime.datetime.strptime(e.split("/")[-1], "%a_%b_%d_%H:%M:%S_%Y") >= INITIAL_DATE]

        if FINAL_DATE:
            episode_paths = [e for e in episode_paths if datetime.datetime.strptime(e.split("/")[-1], "%a_%b_%d_%H:%M:%S_%Y") <= FINAL_DATE]

        print(f"Found {len(episode_paths)} episodes!")
        return {
            'train': episode_paths,
        }
