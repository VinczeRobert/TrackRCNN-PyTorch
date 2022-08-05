import colorsys
from collections import namedtuple

import numpy as np
import munkres
import os

from PIL import Image
from scipy.spatial.distance import cdist
import pycocotools.mask as cocomask

from trackrcnn_kitty.adnotate import adnotate

TrackElement_ = namedtuple("TrackElement", ["track_number", "box", "association_vector", "track_id", "class_", "mask",
                                            "score"])
TrackElement = namedtuple("TrackElement", ["box", "track_id", "class_", "mask", "score"])


def track_for_class(tracker_options_for_class, boxes, scores, association_vectors, classes, masks, class_to_track,
                    start_track_id):
    max_track_id = start_track_id
    all_tracks = []
    active_tracks = []
    munkres_obj = munkres.Munkres()
    for time, (boxes_t, scores_t, association_vectors_t, classes_t, masks_t) in enumerate(
            zip(boxes, scores, association_vectors, classes, masks)):
        detections_t = []
        for box, score, association_vector, class_, mask in \
            zip(boxes_t, scores_t, association_vectors_t, classes_t, masks_t):
            if class_ != class_to_track:
                continue
            if mask is not None and cocomask.area(mask) <= 10:
                continue
            if score >= tracker_options_for_class["confidence_threshold"]:
                detections_t.append((box, association_vector, mask, class_, score))
            else:
                continue
        if len(detections_t) == 0:
            current_tracks = []
        elif len(active_tracks) == 0:
            current_tracks = []
            for det in detections_t:
                current_tracks.append(TrackElement_(track_number=time,
                                                    box=det[0],
                                                    association_vector=det[1],
                                                    mask=det[2],
                                                    class_=det[3],
                                                    track_id=max_track_id,
                                                    score=det[4]))
                max_track_id += 1
        else:
            association_similarities = np.zeros((len(detections_t), len(active_tracks)))

            if tracker_options_for_class["reid_weight"] != 0:
                current_association_vectors = np.array([x[1] for x in detections_t], dtype="float64")
                last_association_vectors = np.array([x.association_vector for x in active_tracks], dtype="float64")

                av_dists = cdist(current_association_vectors, last_association_vectors, "euclidean")
                reid_similarities = tracker_options_for_class["reid_euclidean_scale"] * \
                                  (tracker_options_for_class["reid_euclidean_offset"] - av_dists)
                association_similarities += tracker_options_for_class["reid_weight"] * reid_similarities

            current_tracks = []
            detections_asigned = [False for _ in detections_t]

            cost_matrix = munkres.make_cost_matrix(association_similarities)

            # Assign a very high value for disallowed indices, such that their pairing won't count
            disallow_indices = np.argwhere(association_similarities <=
                                           tracker_options_for_class["association_threshold"])
            for ind in disallow_indices:
                cost_matrix[ind[0]][ind[1]] = 1e9
            indexes = munkres_obj.compute(cost_matrix)
            for row, column in indexes:
                value = cost_matrix[row][column]
                if value == 1e9:
                    continue
                det = detections_t[row]
                track_element = TrackElement_(track_number=time, box=det[0], association_vector=det[1],
                                              mask=det[2], class_=det[3], track_id=active_tracks[column].track_id,
                                              score=det[4])
                current_tracks.append(track_element)
                detections_asigned[row] = True

            for det, assigned in zip(detections_t, detections_asigned):
                if not assigned:
                    current_tracks.append(TrackElement_(track_number=time, box=det[0],
                                                        association_vector=det[1], mask=det[2], class_=det[3],
                                                        track_id=max_track_id, score=det[4]))
                    max_track_id += 1

        all_tracks.append(current_tracks)
        newly_active_ids = {track.track_id for track in current_tracks}
        active_tracks = [track for track in active_tracks
                         if track.track_id not in newly_active_ids and track.track_number >=
                         time - tracker_options_for_class["keep_alive"]]
        active_tracks.extend(current_tracks)

    # remove the association vector values, since they are an implementation detail of the tracker and should not
    # be part of the result
    result = [[TrackElement(box=track.box, track_id=track.track_id, mask=track.mask, class_=track.class_, score=track.score)
               for track in tracks_t] for tracks_t in all_tracks]
    return result


def track_sequence(tracker_options, boxes, scores, association_vectors, classes, masks):
    # tracking will be done in a per-class fashion and the results will be combined in the end
    classes_flat = [c for cs in classes for c in cs]
    unique_classes = np.unique(classes_flat)
    start_track_id = 1
    class_tracks = []
    tracker_options_for_class = {}

    for class_ in unique_classes:
        # for cars
        if class_ == 1:
            tracker_options_for_class["confidence_threshold"] = tracker_options["confidence_threshold_car"]
            tracker_options_for_class["reid_weight"] = tracker_options["reid_weight_car"]
            tracker_options_for_class["association_threshold"] = tracker_options["association_threshold_car"]
            tracker_options_for_class["keep_alive"] = tracker_options["keep_alive_car"]
            tracker_options_for_class["reid_euclidean_offset"] = tracker_options["reid_euclidean_offset_car"]
            tracker_options_for_class["reid_euclidean_scale"] = tracker_options["reid_euclidean_scale_car"]
        elif class_ == 2:
            tracker_options_for_class["confidence_threshold"] = tracker_options["confidence_threshold_pedestrian"]
            tracker_options_for_class["reid_weight"] = tracker_options["reid_weight_pedestrian"]
            tracker_options_for_class["association_threshold"] = tracker_options["association_threshold_pedestrian"]
            tracker_options_for_class["keep_alive"] = tracker_options["keep_alive_pedestrian"]
            tracker_options_for_class["reid_euclidean_pedestrian"] = tracker_options["reid_euclidean_offset_pedestrian"]
            tracker_options_for_class["reid_euclidean_pedestrian"] = tracker_options["reid_euclidean_scale_pedestrian"]
        else:
            assert False, "Unknown class!"

        tracks = track_for_class(tracker_options_for_class, boxes, scores, association_vectors, classes, masks, class_,
                                 start_track_id)
        class_tracks.append(tracks)
        track_ids_flat = [track.track_id for tracks_t in tracks for track in tracks_t]
        track_ids_flat.append(start_track_id)
        start_track_id = max(track_ids_flat) + 1

    n_timesteps = len(boxes)
    tracks_combined = [[] for _ in range(n_timesteps)]
    for tracks_c in class_tracks:
        for t, tracks_c_t in enumerate(tracks_c):
            tracks_combined[t].extend(tracks_c_t)

    return tracks_combined


def make_tracks_disjoint(tracks):
    for frame, objects in enumerate(tracks):
        if len(objects) == 0:
            continue

        objects_sorted = sorted(objects, key=lambda x: x.score, reverse=True)
        objects_disjoint = [objects_sorted[0]]
        used_pixels = objects_sorted[0].mask
        for obj in objects_sorted[1:]:
            new_mask = obj.mask
            if cocomask.area(cocomask.merge([used_pixels, obj.mask], intersect=True)) > 0.0:
                obj_mask_decoded = cocomask.decode(obj.mask)
                used_pixels_decoded = cocomask.decode(used_pixels)
                obj_mask_decoded[np.where(used_pixels_decoded > 0)] = 0
                new_mask = cocomask.encode(obj_mask_decoded)
            used_pixels = cocomask.merge([used_pixels, obj.mask], intersect=False)
            objects_disjoint.append(TrackElement(box=obj.box, track_id=obj.track_id, class_=obj.class_, score=obj.score,
                                                 mask=new_mask))
        tracks[frame] = objects_disjoint

    return tracks


def generate_colors():
    N = 30
    brightness = 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    perm = [15, 13, 25, 12, 19, 8, 22, 24, 29, 17, 28, 20, 2, 27, 11, 26, 21, 4, 3, 18, 9, 5, 14, 1, 16, 0, 23, 7, 6,
            10]
    colors = [colors[idx] for idx in perm]
    return colors


def apply_mask(image, mask, color, alpha=0.5):
    """
    Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c],
                                  image[:, :, c])
    return image


COLORS = [np.random.choice(range(256), size=3) for _ in range(50)]


def visualize_detections(det_boxes, det_classes, det_masks, det_scores, image, ids, save_path):
    colors = [COLORS[id] for id in ids]
    adnotate(image, det_masks, ids, colors, save_path)



def visualize_tracks(sequence_number, tracks, images):
    for t, (track, image) in enumerate(zip(tracks, images)):
        boxes = [te.box for te in track]
        classes = [te.class_ for te in track]
        masks = [cocomask.decode(te.mask) for te in track]
        scores = [1.0 for _ in track]
        ids = [te.track_id for te in track]

        out_folder = os.path.join("tracks_created", sequence_number)
        os.makedirs(out_folder, exist_ok=True)
        out_filename = os.path.join(out_folder, "%06d.jpg" % t)

        visualize_detections(boxes, classes, masks, scores, image, ids, out_filename)

