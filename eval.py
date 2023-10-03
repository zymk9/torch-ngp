import shutil
import os
import h5py
import numpy as np
import imageio
import matplotlib.pyplot as plt
import json
import argparse
from tqdm import tqdm

NYU40_CLASS_NAMES = [
    'wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
    'bookshelf', 'picture', 'counter', 'blinds', 'desk', 'shelves', 'curtain', 'dresser',
    'pillow', 'mirror', 'floormat', 'clothes', 'ceiling', 'books',
    'refrigerator', 'television', 'paper', 'towel', 'showercurtrain', 'box',
    'whiteboard', 'person', 'nightstand', 'toilet', 
    'sink', 'lamp', 'bathtub', 'bag', 'otherstructure', 'otherfurniture', 'otherprop'
]

# mask3d_dir = '/data/bhuai/nerf_rcnn_results/front3d/fcos_vggEF_1k/masks'
mask3d_dir = '/data/bhuai/instance_nerf_data/rcnn_results/front3d_new/masks'

# nyu40 ids used in 3dfront
CLASS_IDS = [3, 4, 5, 6, 7, 10, 14, 32, 35, 39]

with open('front3d_to_nyu40.json', 'r') as f:
    front3d_to_nyu40 = json.load(f)


def add_nyu40_labels(gt_dir):
    for s in os.listdir(gt_dir):
        with open(os.path.join(gt_dir, s, 'gt_instance.json'), 'r') as f:
            meta = json.load(f)

        for obj in meta['instances']:
            nyu40_id = front3d_to_nyu40_id(obj)
            name = NYU40_CLASS_NAMES[nyu40_id - 1]

            obj['nyu40_class'] = name
            obj['class_id'] = nyu40_id

        with open(os.path.join(gt_dir, s, 'gt_instance.json'), 'w') as f:
            json.dump(meta, f, indent=2)


def mIoU_scene(predicted_masks, gt_masks, class_ids):
    '''
    Args:
        predicted_masks: [N, H, W]
        gt_masks: [N, H, W]
    '''
    
    ious = []

    for i in class_ids:
        intersection = np.logical_and(predicted_masks == i, gt_masks == i).sum()
        union = np.logical_or(predicted_masks == i, gt_masks == i).sum()

        if union > 0:
            ious.append(intersection / union)
        else:
            ious.append(None)
    return ious


def PQ_image(predicted_masks, gt_masks, pred_labels, gt_labels):
    '''
    Assume 0 is background, the rest are contiguous instance ids,
    Args:
        predicted_masks: [H, W], in range [0, num_instances]
        gt_masks: [H, W], in range [0, num_instances]
        pred_labels: [num_instances], in CLASS_IDS
        gt_labels: [num_instances], in CLASS_IDS
    '''
    tp_iou_sum = 0
    tp_sum = 0
    fp_sum = 0

    num_pred_instances = pred_labels.shape[0]
    num_gt_instances = gt_labels.shape[0]

    if num_pred_instances == 0:
        return 0, 0, 0, num_gt_instances

    iou_mat = np.zeros((num_pred_instances, num_gt_instances))

    for i in range(num_pred_instances):
        for j in range(num_gt_instances):
            intersection = np.logical_and(predicted_masks == i+1, gt_masks == j+1).sum()
            union = np.logical_or(predicted_masks == i+1, gt_masks == j+1).sum()
            if union > 0:
                iou_mat[i, j] = intersection / union
            else:
                iou_mat[i, j] = 0

    for i in range(num_pred_instances):
        max_iou = iou_mat[i].max()
        argmax_iou = iou_mat[i].argmax()
        if max_iou > 0.5:
            if pred_labels[i] == gt_labels[argmax_iou]:
                tp_iou_sum += max_iou
                tp_sum += 1
            else:
                fp_sum += 1

    gt_max_iou = iou_mat.max(axis=0)
    fn_sum = (gt_max_iou < 0.5).sum()
    
    return tp_iou_sum, tp_sum, fp_sum, fn_sum


def PQ_scene(predicted_masks, gt_masks, pred_labels, gt_labels):
    '''
    Assume 0 is background, the rest are contiguous instance ids,
    Args:
        predicted_masks: [N, H, W], in range [0, num_instances]
        gt_masks: [N, H, W], in range [0, num_instances]
        pred_labels: [num_instances], in CLASS_IDS
        gt_labels: [num_instances], in CLASS_IDS
    '''
    tp_iou_sum = 0
    tp_sum = 0
    fp_sum = 0
    fn_sum = 0

    for i in range(predicted_masks.shape[0]):
        tp_iou, tp, fp, fn = PQ_image(predicted_masks[i], gt_masks[i], pred_labels, gt_labels)
        tp_iou_sum += tp_iou
        tp_sum += tp
        fp_sum += fp
        fn_sum += fn
    
    return tp_iou_sum, tp_sum, fp_sum, fn_sum


def front3d_to_nyu40_id(instance_data):
    cat = instance_data['category']
    sup_cat = instance_data['super-category']
    for item in front3d_to_nyu40:
        if item['front3d_category'] == cat and item['front3d_super_category'] == sup_cat:
            return item['nyu40_id']


def front3d_generate_semantic_masks(scenes, mask_root, meta_root, out_root):
    for scene in scenes:
        mask_dir = os.path.join(mask_root, scene)
        meta_path = os.path.join(meta_root, scene + '.json')
        out_dir = os.path.join(out_root, scene)
        if not os.path.exists(mask_dir):
            print(f'{mask_dir} does not exist')
            continue

        os.makedirs(out_dir, exist_ok=True)

        with open(meta_path, 'r') as f:
            meta = json.load(f)

        instance_id_to_nyu40_id = {}
        for item in meta['instances']:
            # instance_id_to_nyu40_id[item['id']] = item['class_id']
            instance_id_to_nyu40_id[item['id']] = front3d_to_nyu40_id(item)

        print(instance_id_to_nyu40_id)

        masks = os.listdir(mask_dir)
        masks = [m for m in masks if m.endswith('.hdf5')]
        sort_key = lambda x: int(x.split('.')[0])
        masks.sort(key=sort_key)
        for mask_path in masks:
            with h5py.File(os.path.join(mask_dir, mask_path), 'r') as f:
                mask = np.array(f['cp_instance_id_segmaps'][:])

            semantic_mask = np.zeros_like(mask)
            for id in np.unique(mask):
                if id == 0:
                    continue
                semantic_mask[mask == id] = instance_id_to_nyu40_id[id]

            print(np.unique(mask), np.unique(semantic_mask))

            idx = int(mask_path.split('.')[0])
            out_path = os.path.join(out_dir, f'{idx:04d}.npy')
            np.save(out_path, semantic_mask)


def inerf_mask_to_semantic_mask(masks, mask3d):
    '''
    masks: (N, H, W)
    '''
    scores = mask3d['scores']
    labels = mask3d['labels']
    keep = scores > 0.5
    labels = labels[keep]

    semantic_mask = np.zeros_like(masks)
    for i in range(len(masks)):
        for id in np.unique(masks[i]):
            if id == 0:
                continue
            nyu40_id = CLASS_IDS[labels[id - 1] - 1]
            semantic_mask[i][masks[i] == id] = nyu40_id

    return semantic_mask


def save_pair_img(gt_masks, pred_masks, gt_sem, pred_sem, output_dir):
    colors = np.multiply([
        plt.cm.get_cmap('gist_ncar', 37)((i * 7 + 5) % 37)[:3] for i in range(37)
    ], 255).astype(np.uint8)

    for i in range(len(gt_masks)):
        gt_mask = gt_masks[i]
        pred_mask = pred_masks[i]

        gt_img = np.zeros((gt_mask.shape[0], gt_mask.shape[1], 3), dtype=np.uint8)
        pred_img = np.zeros((gt_mask.shape[0], gt_mask.shape[1], 3), dtype=np.uint8)

        for id in np.unique(gt_mask):
            gt_img[gt_mask == id] = colors[id % 37]

        for id in np.unique(pred_mask):
            pred_img[pred_mask == id] = colors[id % 37]

        gt_sem_img = np.zeros((gt_mask.shape[0], gt_mask.shape[1], 3), dtype=np.uint8)
        pred_sem_img = np.zeros((gt_mask.shape[0], gt_mask.shape[1], 3), dtype=np.uint8)

        for id in np.unique(gt_sem[i]):
            gt_sem_img[gt_sem[i] == id] = colors[id % 37]

        for id in np.unique(pred_sem[i]):
            pred_sem_img[pred_sem[i] == id] = colors[id % 37]

        img = np.concatenate([gt_img, pred_img], axis=1)
        img_sem = np.concatenate([gt_sem_img, pred_sem_img], axis=1)
        img = np.concatenate([img, img_sem], axis=0)
        imageio.imwrite(os.path.join(output_dir, f'{i:04d}.png'), img)


def eval(scenes, pred_dir, gt_dir, gt_inst_dir, xform_dir, output_dir, mask3d_dir):
    ious = np.zeros((len(CLASS_IDS) + 1,))
    ious_cnt = np.zeros((len(CLASS_IDS) + 1,))
    print(ious.shape)

    PQ_iou_sum = 0
    PQ_tp_sum = 0
    PQ_fp_sum = 0
    PQ_fn_sum = 0

    for scene in scenes:
        print(scene)
        result_dir = os.path.join(pred_dir, scene, 'results')
        masks = os.listdir(result_dir)
        masks = [m for m in masks if m.endswith('mask.png')]
        masks.sort(key=lambda x: int(x.split('_')[2]))

        os.makedirs(os.path.join(output_dir, scene), exist_ok=True)

        masks_ = []
        for mask in masks:
            mask = imageio.imread(os.path.join(result_dir, mask))
            masks_.append(mask)

        masks = np.stack(masks_, axis=0)

        train_xform = os.path.join(xform_dir, scene, 'train_transforms.json')
        with open(train_xform, 'r') as f:
            train_xform = json.load(f)

        val_xform = os.path.join(xform_dir, scene, 'val_transforms.json')
        with open(val_xform, 'r') as f:
            val_xform = json.load(f)

        mask2d_dir = os.path.join(xform_dir, scene, 'masks_2d')
        mask2d = os.listdir(mask2d_dir)

        val_frames = val_xform['frames']
        val_frames = [f['file_path'] for f in val_frames]
        val_frames = [int(x.split('/')[-1].split('.')[0]) for x in val_frames]
        val_frames = np.array(val_frames)
        val_frames.sort()

        train_frames = train_xform['frames']
        train_frames = [f['file_path'] for f in train_frames]
        train_frames = [int(x.split('/')[-1].split('.')[0]) for x in train_frames]
        train_frames = np.array(train_frames)
        train_frames.sort()

        # print(train_frames.shape, val_frames.shape)

        all_frames = np.concatenate([train_frames, val_frames])
        all_frames.sort()
        valid_frames = []

        for frame in all_frames:
            if os.path.exists(os.path.join(mask2d_dir, f'{frame:04d}.hdf5')):
                valid_frames.append(frame)

        masks_val = []
        valid_val_frames = []
        for frame in val_frames:
            if frame in valid_frames:
                valid_val_frames.append(frame)
                masks_val.append(masks[valid_frames.index(frame)])

        # print(valid_val_frames)

        masks_val = np.stack(masks_val, axis=0)

        mask3d_path = os.path.join(mask3d_dir, scene + '.npz')
        mask3d = np.load(mask3d_path, allow_pickle=True)

        semantic_mask = inerf_mask_to_semantic_mask(masks_val, mask3d)

        for i, frame in enumerate(valid_val_frames):
            imageio.imwrite(os.path.join(output_dir, scene, f'{frame:04d}.png'), semantic_mask[i])

        id2class = {}
        mask3d_scores = mask3d['scores']
        mask3d_labels = mask3d['labels']
        keep = mask3d_scores > 0.5
        mask3d_labels = mask3d_labels[keep]
        mask3d_scores = mask3d_scores[keep]

        # print(f'scores {mask3d_scores}')
        # print(f'labels {mask3d_labels}')

        for i in range(len(mask3d_labels)):
            id2class[i] = CLASS_IDS[mask3d_labels[i] - 1]
        pred_labels = np.array([id2class[i] for i in range(len(id2class))])

        scene_gt = os.path.join(gt_dir, scene)
        gt_masks = os.listdir(scene_gt)
        gt_masks = sorted(gt_masks)

        gt_masks_ = []
        for val in valid_val_frames:
            gt_mask = np.load(os.path.join(scene_gt, gt_masks[val]))
            gt_masks_.append(gt_mask)

        gt_masks = np.stack(gt_masks_, axis=0)

        class_ids = [0] + CLASS_IDS
        scene_iou = mIoU_scene(semantic_mask, gt_masks, class_ids)
        print(scene, 'IoU: ', scene_iou)
        for i in range(len(class_ids)):
            if scene_iou[i] is not None:
                ious[i] += scene_iou[i]
                ious_cnt[i] += 1

        scene_miou = np.mean([iou for iou in scene_iou if iou is not None])
        print(scene, 'mIoU: ', scene_miou)

        # PQ
        with open(os.path.join(gt_inst_dir, scene, 'gt_instance.json'), 'r') as f:
            gt_info = json.load(f)

        gt_id2class = {}
        for ins in gt_info['instances']:
            gt_id2class[ins['id']] = ins['class_id']
        gt_labels = np.array([gt_id2class[x] for x in range(1, len(gt_id2class) + 1)])

        gt_inst_mask_dir = os.path.join(gt_inst_dir, scene, 'masks_2d')
        gt_inst_masks = os.listdir(gt_inst_mask_dir)
        sort_key = lambda x: int(x.split('.')[0])
        gt_inst_masks.sort(key=sort_key)
        gt_inst_masks_ = []
        for gt_inst_mask in gt_inst_masks:
            with h5py.File(os.path.join(gt_inst_mask_dir, gt_inst_mask), 'r') as f:
                gt_inst_mask = f['cp_instance_id_segmaps'][:]
            gt_inst_masks_.append(gt_inst_mask)

        valid_gt_inst_masks = []
        for val in valid_val_frames:
            valid_gt_inst_masks.append(gt_inst_masks_[val])

        gt_inst_masks = np.stack(valid_gt_inst_masks, axis=0)

        PQ_iou, PQ_tp, PQ_fp, PQ_fn = PQ_scene(masks_val, gt_inst_masks, pred_labels, gt_labels)
        print(scene, f'PQ_iou: {PQ_iou} PQ_tp: {PQ_tp} PQ_fp: {PQ_fp} PQ_fn: {PQ_fn}')
        scene_pq = PQ_iou / (PQ_tp + PQ_fp * 0.5 + PQ_fn * 0.5)
        print(scene, f'PQ: {scene_pq}')

        PQ_iou_sum += PQ_iou
        PQ_tp_sum += PQ_tp
        PQ_fp_sum += PQ_fp
        PQ_fn_sum += PQ_fn

        comp_dir = os.path.join(output_dir, scene)
        os.makedirs(comp_dir, exist_ok=True)
        # save_pair_img(gt_masks, semantic_mask, comp_dir)
        save_pair_img(gt_inst_masks, masks_val, gt_masks, semantic_mask, comp_dir)

    keep = ious_cnt > 0

    print(f'iou cnt {ious_cnt}')
    ious = ious[keep]
    ious_cnt = ious_cnt[keep]
    ious = ious / ious_cnt
    print(f'iou {ious}')

    miou = np.mean(ious)
    print(f'miou {miou}')

    PQ = PQ_iou_sum / (PQ_tp_sum + PQ_fp_sum * 0.5 + PQ_fn_sum * 0.5)
    print(f'PQ {PQ}')


def eval_mIoU(pred_sem_root, gt_root, xform_root):
    ious = np.zeros(len(CLASS_IDS) + 1)
    ious_cnt = np.zeros(len(CLASS_IDS) + 1)

    scenes = os.listdir(gt_root)
    scenes.sort()
    for scene in scenes:
        pred_dir = os.path.join(pred_sem_root, scene)
        gt_dir = os.path.join(gt_root, scene)
        
        with open(os.path.join(xform_root, scene, 'val', 'transforms.json'), 'r') as f:
            val_xform = json.load(f)

        frames = val_xform['frames']
        frames = [f['file_path'] for f in frames]
        frames = [x.split('/')[-1].split('.')[0] for x in frames]

        gt_masks = []
        pred_masks = []
        for frame in frames:
            gt_mask = np.load(os.path.join(gt_dir, frame + '.npy'))
            pred_mask = np.load(os.path.join(pred_dir, frame + '.npy'))
            gt_masks.append(gt_mask)
            pred_masks.append(pred_mask)

        gt_masks = np.stack(gt_masks, axis=0)
        pred_masks = np.stack(pred_masks, axis=0)

        class_ids = [0] + CLASS_IDS
        scene_iou = mIoU_scene(pred_masks, gt_masks, class_ids)
        print(scene, scene_iou)
        for i in range(len(class_ids)):
            if scene_iou[i] is not None:
                ious[i] += scene_iou[i]
                ious_cnt[i] += 1

        scene_miou = np.mean([iou for iou in scene_iou if iou is not None])
        print(scene, scene_miou)

    keep = ious_cnt > 0

    print(f'iou cnt {ious_cnt}')
    ious = ious[keep]
    ious_cnt = ious_cnt[keep]
    ious = ious / ious_cnt
    print(f'iou {ious}')

    miou = np.mean(ious)
    print(f'miou {miou}')


def eval_PQ(pred_inst_dir, gt_inst_dir, xform_root):
    PQ_iou_sum = 0
    PQ_tp_sum = 0
    PQ_fp_sum = 0
    PQ_fn_sum = 0

    scenes = os.listdir(gt_inst_dir)
    scenes.sort()

    for scene in scenes:
        with open(os.path.join(xform_root, scene, 'val', 'transforms.json'), 'r') as f:
            val_xform = json.load(f)

        frames = val_xform['frames']
        frames = [f['file_path'] for f in frames]
        frames = [x.split('/')[-1].split('.')[0] for x in frames]

        with open(os.path.join(gt_inst_dir, scene, 'gt_instance.json'), 'r') as f:
            gt_info = json.load(f)

        gt_id2class = {}
        for ins in gt_info['instances']:
            gt_id2class[ins['id']] = ins['class_id']
        gt_labels = np.array([gt_id2class[x] for x in range(1, len(gt_id2class) + 1)])

        gt_inst_mask_dir = os.path.join(gt_inst_dir, scene, 'masks_2d')
        gt_inst_masks = []
        for frame in frames:
            with h5py.File(os.path.join(gt_inst_mask_dir, str(int(frame)) + '.hdf5'), 'r') as f:
                mask = f['cp_instance_id_segmaps'][:]
            gt_inst_masks.append(mask)

        gt_inst_masks = np.stack(gt_inst_masks, axis=0)

        pred_scene_dir = os.path.join(pred_inst_dir, scene)
        pred_labels = []
        pred_masks = []
        for frame in frames:
            with open(os.path.join(pred_scene_dir, frame + '.json'), 'r') as f:
                pred_info = json.load(f)
            labels = np.array([x['nyu40_id'] for x in pred_info])
            pred_labels.append(labels)

            mask = np.load(os.path.join(pred_scene_dir, frame + '.npy'))
            pred_masks.append(mask)

        scene_PQ_iou = 0
        scene_PQ_tp = 0
        scene_PQ_fp = 0
        scene_PQ_fn = 0

        for i, frame in enumerate(frames):
            gt_mask = gt_inst_masks[i]
            pred_mask = pred_masks[i]
            pred_label = pred_labels[i]
        
            PQ_iou, PQ_tp, PQ_fp, PQ_fn = PQ_image(pred_mask, gt_mask, pred_label, gt_labels)
            scene_PQ_iou += PQ_iou
            scene_PQ_tp += PQ_tp
            scene_PQ_fp += PQ_fp
            scene_PQ_fn += PQ_fn

        scene_PQ = scene_PQ_iou / (scene_PQ_tp + scene_PQ_fp * 0.5 + scene_PQ_fn * 0.5)
        print(f'{scene} PQ {scene_PQ} PQ_iou {scene_PQ_iou} PQ_tp {scene_PQ_tp} PQ_fp {scene_PQ_fp} PQ_fn {scene_PQ_fn}')

        PQ_iou_sum += scene_PQ_iou
        PQ_tp_sum += scene_PQ_tp
        PQ_fp_sum += scene_PQ_fp
        PQ_fn_sum += scene_PQ_fn

    PQ = PQ_iou_sum / (PQ_tp_sum + PQ_fp_sum * 0.5 + PQ_fn_sum * 0.5)
    print(f'PQ {PQ}')


def instance_to_semantic(scenes, inst_dir, meta_dir, out_dir):
    for s in scenes:
        inst_dir = os.path.join(inst_dir, s, 'masks_2d')
        with open(os.path.join(meta_dir, s + '.json'), 'r') as f:
            meta = json.load(f)

        instances = meta['instances']
        id2nyu40 = {}
        for ins in instances:
            id2nyu40[ins['id']] = ins['class_id']

        masks = os.listdir(inst_dir)
        sem_map_dir = os.path.join(out_dir, s)
        os.makedirs(sem_map_dir, exist_ok=True)

        for mask_name in masks:
            with h5py.File(os.path.join(inst_dir, mask_name), 'r') as f:
                mask = f['cp_instance_id_segmaps'][:]

                sem_map = np.zeros_like(mask, dtype=np.uint8)
                ids = np.unique(mask)
                for id in ids:
                    if id == 0:
                        continue
                    sem_map[mask == id] = id2nyu40[id]

            mask_idx = int(mask_name.split('.')[0])
            np.save(os.path.join(sem_map_dir, f'{mask_idx:04d}.npy'), sem_map)
