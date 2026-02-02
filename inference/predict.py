"""
Tree identification inference with GPS + CNN combined approach.

Flow:
1. Get photo GPS coordinates
2. Find candidate trees within radius
3. CNN scores candidates
4. Return best match

Usage:
    python predict.py --image path/to/photo.jpg
    python predict.py --image path/to/photo.jpg --lat 59.123 --lon 18.456
"""

import torch
import torch.nn.functional as F
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from pathlib import Path
import pandas as pd
import json
import math
import argparse
from torchvision import transforms
import timm
import torch.nn as nn


class TreeClassifier(nn.Module):
    """EfficientNet-based tree classifier."""

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
        logits = self.classifier(features)
        return logits


def haversine(lat1, lon1, lat2, lon2):
    """Calculate distance in meters between two GPS points."""
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))


def get_exif_gps(image_path):
    """Extract GPS coordinates from image EXIF data."""
    try:
        img = Image.open(image_path)
        exif = img._getexif()
        if not exif:
            return None, None

        gps_info = {}
        for tag_id, value in exif.items():
            tag = TAGS.get(tag_id, tag_id)
            if tag == 'GPSInfo':
                for gps_tag_id, gps_value in value.items():
                    gps_tag = GPSTAGS.get(gps_tag_id, gps_tag_id)
                    gps_info[gps_tag] = gps_value

        if not gps_info:
            return None, None

        def convert_to_degrees(value):
            d, m, s = value
            return float(d) + float(m)/60 + float(s)/3600

        lat = convert_to_degrees(gps_info['GPSLatitude'])
        if gps_info.get('GPSLatitudeRef', 'N') == 'S':
            lat = -lat

        lon = convert_to_degrees(gps_info['GPSLongitude'])
        if gps_info.get('GPSLongitudeRef', 'E') == 'W':
            lon = -lon

        return lat, lon
    except Exception as e:
        return None, None


class TreePredictor:
    """Combined GPS + CNN tree predictor."""

    def __init__(self, model_path, data_dir, device=None, embeddings_path=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.data_dir = Path(data_dir)

        # Load label mappings
        with open(self.data_dir / 'label_encoder_gt.json') as f:
            self.encoder = json.load(f)
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.num_classes = len(self.encoder)

        # Load tree locations
        self._load_tree_locations()

        # Load model
        self._load_model(model_path)

        # Load embeddings if provided
        self.tree_embeddings = None
        if embeddings_path and Path(embeddings_path).exists():
            self._load_embeddings(embeddings_path)

        # Image transform
        self.transform = transforms.Compose([
            transforms.Resize((260, 260)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    def _load_tree_locations(self):
        """Load tree ground truth locations."""
        gt_file = self.data_dir / 'training_data_with_ground_truth.xlsx'
        df = pd.read_excel(gt_file)

        # Get unique tree locations
        self.trees = df.groupby('key').agg({
            'gt_lat': 'first',
            'gt_lon': 'first'
        }).reset_index()

        print(f"Loaded {len(self.trees)} tree locations")

    def _load_model(self, model_path):
        """Load trained model."""
        checkpoint = torch.load(model_path, map_location=self.device)

        # Get backbone from config
        config = checkpoint.get('config', {})
        backbone = config.get('backbone', 'efficientnet_b2')

        self.model = TreeClassifier(self.num_classes, backbone=backbone)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()

        print(f"Loaded model (val_acc: {checkpoint.get('val_acc', 0)*100:.1f}%)")

    def _load_embeddings(self, embeddings_path):
        """Load precomputed tree embeddings."""
        data = torch.load(embeddings_path, map_location=self.device)
        self.tree_embeddings = {k: v.to(self.device) for k, v in data['embeddings'].items()}
        print(f"Loaded embeddings for {len(self.tree_embeddings)} trees")

    def find_nearby_trees(self, lat, lon, radius_m=30):
        """Find trees within radius of given coordinates."""
        candidates = []

        for _, row in self.trees.iterrows():
            dist = haversine(lat, lon, row['gt_lat'], row['gt_lon'])
            if dist <= radius_m:
                candidates.append({
                    'key': row['key'],
                    'lat': row['gt_lat'],
                    'lon': row['gt_lon'],
                    'distance_m': dist,
                    'class_id': self.encoder[row['key']]
                })

        # Sort by distance
        candidates.sort(key=lambda x: x['distance_m'])
        return candidates

    def predict_image(self, image_path):
        """Get CNN predictions for image."""
        img = Image.open(image_path).convert('RGB')
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(img_tensor)
            probs = F.softmax(logits, dim=1)[0]

        return probs.cpu().numpy()

    def get_embedding(self, image_path):
        """Extract normalized embedding from image."""
        img = Image.open(image_path).convert('RGB')
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.model.extract_features(img_tensor)
            features = F.normalize(features, p=2, dim=1)

        return features.squeeze(0)

    def compute_similarity(self, query_embedding, tree_key):
        """Compute cosine similarity between query and tree embedding."""
        if self.tree_embeddings is None or tree_key not in self.tree_embeddings:
            return 0.0
        tree_emb = self.tree_embeddings[tree_key]
        similarity = torch.dot(query_embedding, tree_emb).item()
        return similarity

    def predict(self, image_path, lat=None, lon=None, radius_m=30, top_k=5, distance_weight=0.7):
        """
        Predict tree from image using GPS + CNN hybrid scoring.

        Args:
            image_path: Path to image
            lat, lon: GPS coordinates (extracted from EXIF if not provided)
            radius_m: Search radius for candidate trees
            top_k: Number of results to return
            distance_weight: Weight for distance score (0.7 = 70% distance, 30% CNN)

        Returns:
            dict with prediction results
        """
        image_path = Path(image_path)

        # Get GPS coordinates
        if lat is None or lon is None:
            lat, lon = get_exif_gps(image_path)

        if lat is None:
            return {
                'success': False,
                'error': 'No GPS coordinates available',
                'image': str(image_path)
            }

        # Find candidate trees
        candidates = self.find_nearby_trees(lat, lon, radius_m)

        if not candidates:
            # No trees nearby - fall back to pure CNN
            probs = self.predict_image(image_path)
            top_indices = probs.argsort()[-top_k:][::-1]

            return {
                'success': True,
                'method': 'cnn_only',
                'warning': f'No trees within {radius_m}m of photo location',
                'image': str(image_path),
                'photo_coords': {'lat': lat, 'lon': lon},
                'predictions': [
                    {
                        'rank': i+1,
                        'key': self.decoder[idx],
                        'probability': float(probs[idx]),
                        'class_id': int(idx)
                    }
                    for i, idx in enumerate(top_indices)
                ]
            }

        # Get CNN probabilities
        probs = self.predict_image(image_path)

        # Score candidates with CNN prob
        for c in candidates:
            c['probability'] = float(probs[c['class_id']])

        # Hybrid scoring: combine distance and CNN
        # Distance score: closer = higher (1 - dist/max_dist)
        max_dist = max(c['distance_m'] for c in candidates)
        if max_dist > 0:
            for c in candidates:
                c['dist_score'] = 1 - (c['distance_m'] / (max_dist + 1))
        else:
            for c in candidates:
                c['dist_score'] = 1.0

        # Normalize CNN probs among candidates
        total_prob = sum(c['probability'] for c in candidates)
        if total_prob > 0:
            for c in candidates:
                c['norm_prob'] = c['probability'] / total_prob
        else:
            for c in candidates:
                c['norm_prob'] = 1.0 / len(candidates)

        # Compute hybrid score
        for c in candidates:
            c['hybrid_score'] = distance_weight * c['dist_score'] + (1 - distance_weight) * c['norm_prob']

        # Sort by hybrid score
        candidates.sort(key=lambda x: x['hybrid_score'], reverse=True)

        # Best match
        best = candidates[0]

        return {
            'success': True,
            'method': 'hybrid',
            'distance_weight': distance_weight,
            'image': str(image_path),
            'photo_coords': {'lat': lat, 'lon': lon},
            'search_radius_m': radius_m,
            'candidates_found': len(candidates),
            'prediction': {
                'key': best['key'],
                'hybrid_score': best['hybrid_score'],
                'probability': best['probability'],
                'distance_m': best['distance_m'],
                'coords': {'lat': best['lat'], 'lon': best['lon']}
            },
            'all_candidates': [
                {
                    'rank': i+1,
                    'key': c['key'],
                    'hybrid_score': c['hybrid_score'],
                    'probability': c['probability'],
                    'distance_m': c['distance_m']
                }
                for i, c in enumerate(candidates[:top_k])
            ]
        }


    def predict_embedding(self, image_path, lat=None, lon=None, radius_m=30, top_k=5, distance_weight=0.7):
        """
        Predict tree using GPS + embedding similarity.

        Uses cosine similarity between query image and precomputed tree embeddings.
        """
        image_path = Path(image_path)

        if self.tree_embeddings is None:
            return {'success': False, 'error': 'Embeddings not loaded'}

        # Get GPS coordinates
        if lat is None or lon is None:
            lat, lon = get_exif_gps(image_path)

        if lat is None:
            return {
                'success': False,
                'error': 'No GPS coordinates available',
                'image': str(image_path)
            }

        # Find candidate trees
        candidates = self.find_nearby_trees(lat, lon, radius_m)

        if not candidates:
            return {
                'success': False,
                'error': f'No trees within {radius_m}m',
                'image': str(image_path),
                'photo_coords': {'lat': lat, 'lon': lon}
            }

        # Get query embedding
        query_emb = self.get_embedding(image_path)

        # Compute similarity for each candidate
        for c in candidates:
            c['similarity'] = self.compute_similarity(query_emb, c['key'])

        # Hybrid scoring: distance + similarity
        max_dist = max(c['distance_m'] for c in candidates)
        if max_dist > 0:
            for c in candidates:
                c['dist_score'] = 1 - (c['distance_m'] / (max_dist + 1))
        else:
            for c in candidates:
                c['dist_score'] = 1.0

        # Normalize similarities (shift to 0-1 range)
        min_sim = min(c['similarity'] for c in candidates)
        max_sim = max(c['similarity'] for c in candidates)
        if max_sim > min_sim:
            for c in candidates:
                c['norm_sim'] = (c['similarity'] - min_sim) / (max_sim - min_sim)
        else:
            for c in candidates:
                c['norm_sim'] = 1.0 / len(candidates)

        # Hybrid score
        for c in candidates:
            c['hybrid_score'] = distance_weight * c['dist_score'] + (1 - distance_weight) * c['norm_sim']

        # Sort by hybrid score
        candidates.sort(key=lambda x: x['hybrid_score'], reverse=True)

        best = candidates[0]

        return {
            'success': True,
            'method': 'embedding',
            'distance_weight': distance_weight,
            'image': str(image_path),
            'photo_coords': {'lat': lat, 'lon': lon},
            'search_radius_m': radius_m,
            'candidates_found': len(candidates),
            'prediction': {
                'key': best['key'],
                'hybrid_score': best['hybrid_score'],
                'similarity': best['similarity'],
                'distance_m': best['distance_m'],
                'coords': {'lat': best['lat'], 'lon': best['lon']}
            },
            'all_candidates': [
                {
                    'rank': i+1,
                    'key': c['key'],
                    'hybrid_score': c['hybrid_score'],
                    'similarity': c['similarity'],
                    'distance_m': c['distance_m']
                }
                for i, c in enumerate(candidates[:top_k])
            ]
        }


def main():
    parser = argparse.ArgumentParser(description='Predict tree from image')
    parser.add_argument('--image', type=str, required=True, help='Path to image')
    parser.add_argument('--lat', type=float, help='Latitude (uses EXIF if not provided)')
    parser.add_argument('--lon', type=float, help='Longitude (uses EXIF if not provided)')
    parser.add_argument('--radius', type=float, default=30, help='Search radius in meters')
    parser.add_argument('--model', type=str, default=r'E:\tree_id_2.0\models\model_seed42.pt')
    parser.add_argument('--data', type=str, default=r'E:\tree_id_2.0\data')
    args = parser.parse_args()

    # Create predictor
    predictor = TreePredictor(args.model, args.data)

    # Predict
    result = predictor.predict(
        args.image,
        lat=args.lat,
        lon=args.lon,
        radius_m=args.radius
    )

    # Print result
    print("\n" + "="*50)
    if result['success']:
        if result['method'] == 'hybrid':
            print(f"PREDICTION: {result['prediction']['key']}")
            print(f"Hybrid score: {result['prediction']['hybrid_score']:.3f}")
            print(f"CNN prob: {result['prediction']['probability']*100:.1f}%")
            print(f"Distance: {result['prediction']['distance_m']:.1f}m")
            print(f"Candidates within {result['search_radius_m']}m: {result['candidates_found']}")
            print(f"\nAll candidates (weight: {result['distance_weight']*100:.0f}% dist, {(1-result['distance_weight'])*100:.0f}% CNN):")
            for c in result['all_candidates']:
                print(f"  {c['rank']}. {c['key']} - score:{c['hybrid_score']:.3f} prob:{c['probability']*100:.1f}% dist:{c['distance_m']:.1f}m")
        elif result['method'] == 'cnn_only':
            print(f"WARNING: {result.get('warning', 'Using CNN only')}")
            print("\nTop predictions (CNN only):")
            for p in result['predictions']:
                print(f"  {p['rank']}. {p['key']} - {p['probability']*100:.1f}%")
    else:
        print(f"ERROR: {result['error']}")
    print("="*50)


if __name__ == '__main__':
    main()
