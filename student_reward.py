"""Optional E5 reward only. No GRPO training results are claimed."""
from scripts.score import iou_xyxy, parse_output


def iou_reward(completion, gt_boxes):
    """Same-class greedy IoU matching; gt_boxes use label + bbox_2d on 0–1000."""
    predictions = parse_output(completion)
    if predictions is None:
        return 0.0
    if not predictions and not gt_boxes:
        return 1.0
    candidates = []
    for pi, prediction in enumerate(predictions):
        for gi, truth in enumerate(gt_boxes):
            if prediction['label'] == truth['label']:
                overlap = iou_xyxy(prediction['bbox'], truth['bbox_2d'])
                if overlap >= 0.5:
                    candidates.append((overlap, pi, gi))
    used_predictions, used_truths, total = set(), set(), 0.0
    for overlap, pi, gi in sorted(candidates, reverse=True):
        if pi not in used_predictions and gi not in used_truths:
            used_predictions.add(pi)
            used_truths.add(gi)
            total += overlap
    return 0.2 + 0.8 * total / max(len(predictions), len(gt_boxes), 1)
