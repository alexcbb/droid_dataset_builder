import tqdm
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
import wandb
import tensorflow as tf

WANDB_ENTITY = None
WANDB_PROJECT = 'vis_rlds'

DATASET_PATH = "/home/panda/tensorflow_datasets"
DATASET_NAME = "liris_pnp_orange"

if WANDB_ENTITY is not None:
    render_wandb = True
    wandb.init(entity=WANDB_ENTITY,
               project=WANDB_PROJECT)
else:
    render_wandb = False


# create TF dataset
print(f"Visualizing data from dataset: {DATASET_NAME}")
ds = tfds.load(DATASET_NAME, data_dir=DATASET_PATH, split='train')
ds = ds.shuffle(10)

# visualize episodes
for i, episode in enumerate(ds.take(10)):
    images_ext_1, images_ext_2, images_wrist = [], [], []
    for step in episode['steps']:
        images_ext_1.append(step['observation']['exterior_image_1_left'].numpy())
        images_wrist.append(step['observation']['wrist_image_left'].numpy())

    image_strip_ext_1 = np.concatenate(images_ext_1[::4], axis=1)
    image_strip_wrist = np.concatenate(images_wrist[::4], axis=1)
    image_strip = np.concatenate((image_strip_ext_1, image_strip_wrist), axis=0)
    caption = step['language_instruction'].numpy().decode() + ' (temp. downsampled 4x)'

    if render_wandb:
        wandb.log({f'image_{i}': wandb.Image(image_strip, caption=caption)})
    else:
        plt.figure()
        plt.imshow(image_strip)
        plt.title(caption)

# visualize action and state statistics
actions, states = [], []
for episode in tqdm.tqdm(ds.take(500)):
    for step in episode['steps']:
        actions.append(step['action'].numpy())
        states.append(step['observation']['cartesian_position'].numpy())
actions = np.array(actions)
states = np.array(states)
action_mean = actions.mean(0)
state_mean = states.mean(0)

def vis_stats(vector, vector_mean, tag):
    assert len(vector.shape) == 2
    assert len(vector_mean.shape) == 1
    assert vector.shape[1] == vector_mean.shape[0]

    n_elems = vector.shape[1]
    fig = plt.figure(tag, figsize=(5*n_elems, 5))
    for elem in range(n_elems):
        plt.subplot(1, n_elems, elem+1)
        plt.hist(vector[:, elem], bins=20)
        plt.title(vector_mean[elem])

    if render_wandb:
        wandb.log({tag: wandb.Image(fig)})

vis_stats(actions, action_mean, 'action_stats')
vis_stats(states, state_mean, 'state_stats')

if not render_wandb:
    plt.show()


