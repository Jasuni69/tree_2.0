"""
Evaluate GPS+CNN combined approach on training data.

Compares:
1. CNN only (pure classification)
2. GPS only (nearest tree)
3. GPS + CNN (filter by GPS, rank by CNN)
"""

import torch
import torch.nn.functional as F
from PIL import Image
from pathlib import Path
import pandas as pd
import json
import math
from tqdm import tqdm
from torchvision import transforms
import timm
import torch.nn as nn
import numpy as np


class TreeClassifier(nn.Module):
    def __init__(self, num_classes, backbone='efficientnet_b2'):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=False, num_classes=0)
        self.feature_dim = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.feature_dim, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

    def extract_features(self, x):
        return self.backbone(x)


def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))


def evaluate(model_path, data_dir, image_base, radius_m=30, sample_size=None, embeddings_path=None, input_file='training_data_with_ground_truth.xlsx'):
    """Evaluate all methods including embedding-based."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_dir = Path(data_dir)
    image_base = Path(image_base)

    # Load embeddings if available
    tree_embeddings = None
    tree_prototypes = None
    if embeddings_path and Path(embeddings_path).exists():
        emb_data = torch.load(embeddings_path, map_location=device)
        tree_embeddings = {k: v.to(device) for k, v in emb_data['embeddings'].items()}
        print(f"Loaded embeddings for {len(tree_embeddings)} trees")

        # Load prototypes if available (v2 format)
        if 'prototypes' in emb_data:
            tree_prototypes = {k: [p.to(device) for p in protos]
                              for k, protos in emb_data['prototypes'].items()}
            avg_protos = np.mean([len(p) for p in tree_prototypes.values()])
            print(f"Loaded prototypes (avg {avg_protos:.1f} per tree)")

    # Load data
    with open(data_dir / 'label_encoder_gt.json') as f:
        encoder = json.load(f)
    decoder = {v: k for k, v in encoder.items()}
    num_classes = len(encoder)

    df = pd.read_excel(data_dir / input_file)
    print(f"Loaded data from: {input_file}")

    # Get tree locations
    trees = df.groupby('key').agg({
        'gt_lat': 'first',
        'gt_lon': 'first'
    }).reset_index()
    tree_coords = {row['key']: (row['gt_lat'], row['gt_lon']) for _, row in trees.iterrows()}

    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint.get('config', {})
    backbone = config.get('backbone', 'efficientnet_b2')

    model = TreeClassifier(num_classes, backbone=backbone)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"Model val_acc: {checkpoint.get('val_acc', 0)*100:.1f}%")

    # Transform - base (no augmentation)
    transform = transforms.Compose([
        transforms.Resize((260, 260)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # TTA transforms (5 augmentations)
    tta_transforms = [
        transform,  # Original
        transforms.Compose([
            transforms.Resize((260, 260)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        transforms.Compose([
            transforms.Resize((292, 292)),
            transforms.CenterCrop(260),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        transforms.Compose([
            transforms.Resize((260, 260)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        transforms.Compose([
            transforms.Resize((280, 280)),
            transforms.CenterCrop(260),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
    ]

    # Sample if needed
    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42)
        print(f"Evaluating on {len(df)} samples")
    else:
        print(f"Evaluating on {len(df)} photos")

    # Track results
    results = {
        'cnn_only': {'correct': 0, 'total': 0},
        'gps_only': {'correct': 0, 'total': 0},
        'gps_cnn': {'correct': 0, 'total': 0},
        'gps_cnn_top3': {'correct': 0, 'total': 0},
        'hybrid_0.5': {'correct': 0, 'total': 0},  # 50% distance, 50% CNN
        'hybrid_0.7': {'correct': 0, 'total': 0},  # 70% distance, 30% CNN
        'hybrid_0.8': {'correct': 0, 'total': 0},  # 80% distance, 20% CNN
        'hybrid_0.9': {'correct': 0, 'total': 0},  # 90% distance, 10% CNN
        'emb_only': {'correct': 0, 'total': 0},    # Embedding similarity only
        'emb_hybrid_0.5': {'correct': 0, 'total': 0},  # 50% dist + 50% embedding
        'emb_hybrid_0.7': {'correct': 0, 'total': 0},  # 70% dist + 30% embedding
        # New methods
        'tta_emb_only': {'correct': 0, 'total': 0},   # TTA embedding
        'tta_emb_hybrid_0.5': {'correct': 0, 'total': 0},  # TTA + hybrid
        'ensemble_0.5': {'correct': 0, 'total': 0},   # 50% emb + 50% cnn
        'ensemble_hybrid': {'correct': 0, 'total': 0},  # dist + emb + cnn combined
        'exp_dist_emb': {'correct': 0, 'total': 0},   # Exponential distance + emb
        # Prototype methods
        'proto_only': {'correct': 0, 'total': 0},     # Max similarity to any prototype
        'proto_hybrid_0.5': {'correct': 0, 'total': 0},  # Distance + prototype
    }

    errors = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc='Evaluating'):
        img_path = image_base / row['image_path']
        true_key = row['key']
        true_class = encoder[true_key]
        photo_lat, photo_lon = row['photo_lat'], row['photo_lon']

        # Load and predict
        try:
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = model(img_tensor)
                probs = F.softmax(logits, dim=1)[0].cpu().numpy()
                # Extract embedding for similarity-based methods
                if tree_embeddings:
                    features = model.extract_features(img_tensor)
                    query_emb = F.normalize(features, p=2, dim=1).squeeze(0)

                    # TTA: extract embeddings for all augmented versions
                    tta_embs = []
                    for tta_t in tta_transforms:
                        tta_tensor = tta_t(img).unsqueeze(0).to(device)
                        tta_feat = model.extract_features(tta_tensor)
                        tta_embs.append(F.normalize(tta_feat, p=2, dim=1))
                    # Average TTA embeddings
                    tta_emb = torch.cat(tta_embs, dim=0).mean(dim=0)
                    tta_emb = F.normalize(tta_emb.unsqueeze(0), p=2, dim=1).squeeze(0)
        except Exception as e:
            continue

        # 1. CNN only
        cnn_pred = probs.argmax()
        cnn_correct = (cnn_pred == true_class)
        results['cnn_only']['correct'] += cnn_correct
        results['cnn_only']['total'] += 1

        # 2. Find candidates within radius
        candidates = []
        for key, (t_lat, t_lon) in tree_coords.items():
            dist = haversine(photo_lat, photo_lon, t_lat, t_lon)
            if dist <= radius_m:
                candidates.append({
                    'key': key,
                    'class_id': encoder[key],
                    'distance': dist,
                    'prob': probs[encoder[key]]
                })

        if not candidates:
            # No candidates - skip GPS methods
            continue

        # 3. GPS only (nearest tree)
        candidates.sort(key=lambda x: x['distance'])
        gps_pred = candidates[0]['key']
        gps_correct = (gps_pred == true_key)
        results['gps_only']['correct'] += gps_correct
        results['gps_only']['total'] += 1

        # 4. GPS + CNN (highest prob among candidates)
        candidates.sort(key=lambda x: x['prob'], reverse=True)
        gps_cnn_pred = candidates[0]['key']
        gps_cnn_correct = (gps_cnn_pred == true_key)
        results['gps_cnn']['correct'] += gps_cnn_correct
        results['gps_cnn']['total'] += 1

        # 5. GPS + CNN top-3
        top3_keys = [c['key'] for c in candidates[:3]]
        gps_cnn_top3_correct = (true_key in top3_keys)
        results['gps_cnn_top3']['correct'] += gps_cnn_top3_correct
        results['gps_cnn_top3']['total'] += 1

        # 6. Hybrid scoring - combine distance and CNN
        # Normalize distance: closer = higher score (1 - dist/radius)
        # Normalize probs: already 0-1
        max_dist = max(c['distance'] for c in candidates)
        if max_dist > 0:
            for c in candidates:
                c['dist_score'] = 1 - (c['distance'] / (max_dist + 1))
        else:
            for c in candidates:
                c['dist_score'] = 1.0

        # Normalize CNN probs among candidates
        total_prob = sum(c['prob'] for c in candidates)
        if total_prob > 0:
            for c in candidates:
                c['norm_prob'] = c['prob'] / total_prob
        else:
            for c in candidates:
                c['norm_prob'] = 1.0 / len(candidates)

        # Test different weights
        for weight in [0.5, 0.7, 0.8, 0.9]:
            for c in candidates:
                c['hybrid_score'] = weight * c['dist_score'] + (1 - weight) * c['norm_prob']
            candidates.sort(key=lambda x: x['hybrid_score'], reverse=True)
            hybrid_pred = candidates[0]['key']
            hybrid_correct = (hybrid_pred == true_key)
            results[f'hybrid_{weight}']['correct'] += hybrid_correct
            results[f'hybrid_{weight}']['total'] += 1

        # 7. Embedding-based methods
        if tree_embeddings:
            # Compute similarity for each candidate
            for c in candidates:
                if c['key'] in tree_embeddings:
                    c['similarity'] = torch.dot(query_emb, tree_embeddings[c['key']]).item()
                else:
                    c['similarity'] = 0.0

            # Embedding only (highest similarity)
            candidates.sort(key=lambda x: x['similarity'], reverse=True)
            emb_pred = candidates[0]['key']
            emb_correct = (emb_pred == true_key)
            results['emb_only']['correct'] += emb_correct
            results['emb_only']['total'] += 1

            # Normalize similarities for hybrid
            min_sim = min(c['similarity'] for c in candidates)
            max_sim = max(c['similarity'] for c in candidates)
            if max_sim > min_sim:
                for c in candidates:
                    c['norm_sim'] = (c['similarity'] - min_sim) / (max_sim - min_sim)
            else:
                for c in candidates:
                    c['norm_sim'] = 1.0 / len(candidates)

            # Embedding hybrid scores
            for weight in [0.5, 0.7]:
                for c in candidates:
                    c['emb_hybrid'] = weight * c['dist_score'] + (1 - weight) * c['norm_sim']
                candidates.sort(key=lambda x: x['emb_hybrid'], reverse=True)
                emb_hybrid_pred = candidates[0]['key']
                emb_hybrid_correct = (emb_hybrid_pred == true_key)
                results[f'emb_hybrid_{weight}']['correct'] += emb_hybrid_correct
                results[f'emb_hybrid_{weight}']['total'] += 1

            # === NEW METHODS ===

            # 8. TTA embedding similarity
            for c in candidates:
                if c['key'] in tree_embeddings:
                    c['tta_similarity'] = torch.dot(tta_emb, tree_embeddings[c['key']]).item()
                else:
                    c['tta_similarity'] = 0.0

            candidates.sort(key=lambda x: x['tta_similarity'], reverse=True)
            tta_emb_pred = candidates[0]['key']
            results['tta_emb_only']['correct'] += (tta_emb_pred == true_key)
            results['tta_emb_only']['total'] += 1

            # 9. TTA + hybrid (distance + TTA embedding)
            min_tta_sim = min(c['tta_similarity'] for c in candidates)
            max_tta_sim = max(c['tta_similarity'] for c in candidates)
            if max_tta_sim > min_tta_sim:
                for c in candidates:
                    c['norm_tta_sim'] = (c['tta_similarity'] - min_tta_sim) / (max_tta_sim - min_tta_sim)
            else:
                for c in candidates:
                    c['norm_tta_sim'] = 1.0 / len(candidates)

            for c in candidates:
                c['tta_hybrid'] = 0.5 * c['dist_score'] + 0.5 * c['norm_tta_sim']
            candidates.sort(key=lambda x: x['tta_hybrid'], reverse=True)
            results['tta_emb_hybrid_0.5']['correct'] += (candidates[0]['key'] == true_key)
            results['tta_emb_hybrid_0.5']['total'] += 1

            # 10. Ensemble: combine embedding + classification
            for c in candidates:
                c['ensemble'] = 0.5 * c['norm_sim'] + 0.5 * c['norm_prob']
            candidates.sort(key=lambda x: x['ensemble'], reverse=True)
            results['ensemble_0.5']['correct'] += (candidates[0]['key'] == true_key)
            results['ensemble_0.5']['total'] += 1

            # 11. Full ensemble: distance + embedding + classification
            for c in candidates:
                c['full_ensemble'] = 0.4 * c['dist_score'] + 0.4 * c['norm_sim'] + 0.2 * c['norm_prob']
            candidates.sort(key=lambda x: x['full_ensemble'], reverse=True)
            results['ensemble_hybrid']['correct'] += (candidates[0]['key'] == true_key)
            results['ensemble_hybrid']['total'] += 1

            # 12. Exponential distance decay + embedding
            sigma = 10.0  # distance decay parameter
            for c in candidates:
                c['exp_dist'] = math.exp(-c['distance'] / sigma)
            # Normalize exp distances
            total_exp = sum(c['exp_dist'] for c in candidates)
            for c in candidates:
                c['norm_exp_dist'] = c['exp_dist'] / total_exp if total_exp > 0 else 1.0 / len(candidates)
            # Combine with embedding
            for c in candidates:
                c['exp_emb'] = 0.5 * c['norm_exp_dist'] + 0.5 * c['norm_sim']
            candidates.sort(key=lambda x: x['exp_emb'], reverse=True)
            results['exp_dist_emb']['correct'] += (candidates[0]['key'] == true_key)
            results['exp_dist_emb']['total'] += 1

            # 13. Prototype-based: max similarity to any prototype
            if tree_prototypes:
                for c in candidates:
                    if c['key'] in tree_prototypes:
                        # Max similarity to any prototype
                        proto_sims = [torch.dot(query_emb, p).item() for p in tree_prototypes[c['key']]]
                        c['proto_sim'] = max(proto_sims)
                    else:
                        c['proto_sim'] = c['similarity']  # Fallback to mean

                # Proto only
                candidates.sort(key=lambda x: x['proto_sim'], reverse=True)
                results['proto_only']['correct'] += (candidates[0]['key'] == true_key)
                results['proto_only']['total'] += 1

                # Proto hybrid with distance
                min_proto = min(c['proto_sim'] for c in candidates)
                max_proto = max(c['proto_sim'] for c in candidates)
                if max_proto > min_proto:
                    for c in candidates:
                        c['norm_proto'] = (c['proto_sim'] - min_proto) / (max_proto - min_proto)
                else:
                    for c in candidates:
                        c['norm_proto'] = 1.0 / len(candidates)

                for c in candidates:
                    c['proto_hybrid'] = 0.5 * c['dist_score'] + 0.5 * c['norm_proto']
                candidates.sort(key=lambda x: x['proto_hybrid'], reverse=True)
                results['proto_hybrid_0.5']['correct'] += (candidates[0]['key'] == true_key)
                results['proto_hybrid_0.5']['total'] += 1

        # Track errors for analysis
        if not gps_cnn_correct:
            errors.append({
                'image': str(img_path),
                'true_key': true_key,
                'predicted': gps_cnn_pred,
                'true_prob': probs[true_class],
                'pred_prob': candidates[0]['prob'],
                'candidates': len(candidates),
                'true_distance': haversine(photo_lat, photo_lon, *tree_coords[true_key])
            })

    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)

    for method, data in results.items():
        if data['total'] > 0:
            acc = data['correct'] / data['total'] * 100
            print(f"{method:15} : {acc:5.1f}% ({data['correct']}/{data['total']})")

    # Error analysis
    if errors:
        print(f"\n=== Error Analysis ({len(errors)} errors) ===")
        print(f"Avg candidates when wrong: {np.mean([e['candidates'] for e in errors]):.1f}")
        print(f"Avg true_prob when wrong: {np.mean([e['true_prob'] for e in errors])*100:.2f}%")
        print(f"Avg pred_prob when wrong: {np.mean([e['pred_prob'] for e in errors])*100:.2f}%")

    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=r'E:\tree_id_2.0\models\model_seed42.pt')
    parser.add_argument('--data', default=r'E:\tree_id_2.0\data')
    parser.add_argument('--images', default=r'E:\tree_id_2.0\images')
    parser.add_argument('--embeddings', default=r'E:\tree_id_2.0\models\tree_embeddings.pt')
    parser.add_argument('--radius', type=float, default=30)
    parser.add_argument('--sample', type=int, default=None, help='Sample size for quick test')
    parser.add_argument('--input', default='training_data_with_ground_truth.xlsx', help='Input file name')
    args = parser.parse_args()

    evaluate(args.model, args.data, args.images, args.radius, args.sample, args.embeddings, args.input)
