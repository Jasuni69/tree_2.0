# Next Training Plan: Maximize Same-Address Tree Discrimination

## Problem Summary

Current system: ConvNeXt-Base, 224x224 input, 1024-dim embedding, ArcFace + BatchHard triplet loss.

Key findings from error analysis:
- 39.1% error rate on non-trivial tests (617/1579 errors)
- 92% of errors have margin <0.005 (embeddings nearly identical)
- More candidates = worse: 2 trees = 83% acc, 27 trees = 19% acc
- Top 10 worst addresses = 42% of all errors
- 52 trees ALWAYS fail (0% acc) despite 17-25 training photos
- Rank 2 in 40% of errors (correct tree is second choice)

Core issue: Model cannot discriminate between similar trees at same address. Embeddings collapse when trees share location and similar appearance.

---

## Ranked Proposals by Expected ROI

### **TIER 1: High Impact, Medium Effort**

#### 1. Address-Aware ArcFace Margin (HIGHEST PRIORITY)
**Problem addressed:** 92% of errors have margin <0.005. Model doesn't push same-address classes far enough apart.

**Approach:**
- Modify SubCenterArcFace to accept per-sample margins
- Increase margin for same-address tree pairs from 0.5 to 0.7-0.8
- Keep margin at 0.5 for different-address trees
- Implementation: precompute address-to-classes mapping, pass dynamic margins to loss

**Expected impact:**
- Force larger angular separation for confusable classes
- Directly targets 92% of errors with tiny margins
- Should improve worst-case addresses (42% of errors)

**Implementation difficulty:** Medium
- Modify loss function (50 lines)
- Precompute address groups at init
- Pass margin tensor instead of scalar

**Expected gain:** +10-15% accuracy on same-address discrimination

---

#### 2. Higher Resolution Input (384x384)
**Problem addressed:** Trees differ in fine-grained details (bark texture, branch patterns). 224x224 may lose critical info.

**Approach:**
- Increase to 384x384 (not 448 yet - memory constraints)
- Update transforms: Resize(400), RandomResizedCrop(384) for train
- ConvNeXt-Base can handle this (fully convolutional after stem)
- Memory cost: ~3x (batch might drop from P=16,K=4 to P=12,K=4)

**Expected impact:**
- More discriminative features from bark/texture/fine structure
- Helps 52 "always fail" trees if they have unique fine details
- Better performance on crowded addresses (27 candidates)

**Implementation difficulty:** Low
- Change 3 numbers in dataset_metric.py
- Adjust batch size for GPU memory
- No model changes needed

**Tradeoffs:**
- 3x memory usage (may need smaller batch)
- 1.5x slower training
- Diminishing returns beyond 384 for ConvNeXt-Base (receptive field limits)

**Expected gain:** +8-12% accuracy, especially on high-candidate addresses

---

#### 3. Hard Address Mining in Sampling
**Problem addressed:** Top 10 addresses = 42% of errors. Model needs more exposure to worst cases.

**Approach:**
- Track per-address error rates during validation
- Oversample worst addresses by 2-3x in training batches
- Curriculum: start with uniform sampling (epochs 1-10), then ramp up hard mining (epochs 11+)
- Keep AddressAwarePKSampler structure but bias address selection

**Expected impact:**
- More gradient updates on hardest cases
- Prevents "giving up" on worst addresses
- Complements address-aware margins

**Implementation difficulty:** Medium
- Add address error tracking in validation
- Modify AddressAwarePKSampler to accept sampling weights
- Save/load weights across epochs

**Expected gain:** +5-10% reduction in worst-address errors

---

### **TIER 2: Medium Impact, Variable Effort**

#### 4. Multi-Similarity Loss (Replace/Augment Triplet)
**Problem addressed:** BatchHard only considers hardest pos/neg per anchor. Misses other informative pairs.

**Approach:**
- Replace BatchHardTripletLoss with Multi-Similarity Loss
- MS-Loss uses weighted contributions from multiple positives/negatives
- Better gradient flow for moderate-difficulty pairs (not just hardest)
- Keep ArcFace, replace triplet term

**Expected impact:**
- Smoother optimization (less sensitive to single hard pair)
- May help "rank 2" errors (40% of failures) by pushing more negatives away
- Better for imbalanced scenarios (trees with 3 photos vs 25)

**Implementation difficulty:** Medium-High
- Implement MS-Loss from scratch (100 lines)
- Tune hyperparameters (alpha, beta, lambda, margin)
- Risk: may destabilize training if tuned wrong

**Expected gain:** +3-8% accuracy, especially on rank 2 errors

---

#### 5. Advanced Augmentation for Trees
**Problem addressed:** Model overfits to lighting/seasonal conditions. Same tree in different seasons looks different.

**Approach:**
- **Seasonal simulation:** aggressive ColorJitter (brightness ±0.5, saturation ±0.4)
- **Perspective/angle variation:** RandomPerspective(distortion=0.3)
- **Partial occlusion:** RandomErasing(p=0.3) + Cutout
- **NO CutMix/MixUp:** these hurt metric learning (mixes labels)
- **Keep current:** rotation, flip, blur

**Expected impact:**
- Better invariance to capture conditions
- Helps trees with seasonal variation in training set
- Reduces overfitting on small-sample trees (3-5 photos)

**Implementation difficulty:** Low
- Add 3 transforms to dataset_metric.py
- Tune probabilities

**Expected gain:** +3-5% accuracy, helps generalization

---

#### 6. Larger Embedding Dimension (1024 → 1536)
**Problem addressed:** 1024 dims may not have capacity for fine-grained separation of 500+ trees with high within-address similarity.

**Approach:**
- Increase embedding_head final layer to 1536
- More "room" for model to spread embeddings
- Update ArcFace in_features to 1536

**Expected impact:**
- Potentially reduces embedding collapse (margin <0.005)
- Helps when many trees at same address (27 candidates)

**Implementation difficulty:** Low
- Change 2 numbers
- Rerun training from scratch (can't transfer 1024→1536 head)

**Tradeoffs:**
- Slower inference (50% more compute for similarity)
- More parameters to overfit (need strong regularization)
- May not help if features are the issue, not capacity

**Expected gain:** +2-5% if capacity is bottleneck, else minimal

---

### **TIER 3: Lower Impact or High Risk**

#### 7. ConvNeXt-Large Backbone
**Problem addressed:** Larger model = more capacity for fine-grained features.

**Approach:**
- Swap convnext_base (1024 dim) → convnext_large (1536 dim)
- Pretrained on ImageNet
- Update BACKBONE_DIMS in model_metric.py

**Expected impact:**
- More expressive features
- May help 52 "always fail" trees

**Implementation difficulty:** Low (change 1 line)

**Tradeoffs:**
- 3x slower training
- 2x more memory (batch size drops significantly)
- Needs much more data to avoid overfitting (we have ~8k photos)
- Diminishing returns vs resolution increase

**Expected gain:** +3-6% but at high computational cost. **Not recommended unless other methods fail.**

---

#### 8. Multiple Embedding Heads (Coarse + Fine)
**Problem addressed:** Single embedding must encode both coarse (species) and fine (individual) features.

**Approach:**
- Two parallel heads: coarse_emb (512 dim) + fine_emb (512 dim) = 1024 total
- Coarse head trained with standard ArcFace (all classes)
- Fine head trained with address-aware ArcFace (only same-address classes)
- Concatenate for final embedding

**Expected impact:**
- Hierarchical representation
- Fine head focuses only on within-address discrimination

**Implementation difficulty:** High
- Major model architecture change
- Need separate loss terms, careful weighting
- Validation logic more complex

**Tradeoffs:**
- High complexity (lots of hyperparameters)
- Risk of one head dominating
- Unclear if better than single head with address-aware loss

**Expected gain:** +5-10% if tuned perfectly, but high risk of failure. **Defer to future work.**

---

#### 9. Attention Mechanisms for Discriminative Regions
**Problem addressed:** Model may attend to background/irrelevant features instead of discriminative bark/branch patterns.

**Approach:**
- Add spatial attention module before pooling (CBAM or SE block)
- Forces model to weight important regions (trunk, unique branches)
- Insert after backbone, before GeM pooling

**Expected impact:**
- Better feature localization
- May help when trees differ only in small regions

**Implementation difficulty:** Medium
- Add attention module (30 lines)
- Risk: may overfit to training set backgrounds

**Expected gain:** +2-4%. **Lower priority than resolution/margins.**

---

### **TIER 4: Not Recommended**

#### 10. Circle Loss
**Problem addressed:** Alternative to ArcFace+Triplet.

**Why skip:**
- Similar to MS-Loss but harder to tune
- Current ArcFace works well (not the bottleneck)
- Adds complexity without clear win

---

#### 11. Semi-Hard Mining (vs Hardest)
**Problem addressed:** Hardest negatives may be outliers/label noise.

**Why skip:**
- Our data is clean (manually labeled)
- Semi-hard may slow convergence
- BatchHard working fine (loss is stable in current training)

---

## Recommended Training Run Configuration

### **Priority 1: Combined Quick Wins**

Implement Tier 1 #1 + #2 + #3 together:
- 384x384 resolution
- Address-aware ArcFace margins (0.7 for same-address, 0.5 for different)
- Hard address mining (oversample worst 20 addresses by 3x after epoch 10)

**Expected total gain:** +20-30% accuracy on same-address cases

**Training config changes:**
```
Resolution: 384x384 (from 224)
Batch: P=12, K=4 (from P=16, K=4) - adjust for memory
ArcFace margin: dynamic (0.7 same-address, 0.5 different)
Triplet: keep BatchHard at 0.5
Sampling: address error-weighted after epoch 10
Epochs: 80 (from 60) - more data per epoch with hard mining
```

---

### **Priority 2: If Priority 1 Insufficient**

Add Tier 2 #4 + #5:
- Replace triplet with Multi-Similarity Loss
- Enhanced augmentation (perspective, stronger color, more erasing)

**Expected additional gain:** +5-10%

---

### **Priority 3: Nuclear Option**

If still failing on worst addresses:
- Increase embedding dim to 1536 (Tier 2 #6)
- Add spatial attention (Tier 3 #9)
- Consider manual feature engineering for worst 52 trees (human-in-loop labeling of discriminative regions)

---

## Implementation Checklist (Priority 1)

1. **Resolution change** (15 min)
   - dataset_metric.py: Resize(400), RandomResizedCrop(384), CenterCrop(384)
   - train_metric.py: BATCH_P=12 (test GPU memory)

2. **Address-aware margins** (2 hours)
   - Precompute address groups from encoder in train_metric.py
   - Modify SubCenterArcFace.forward() to accept margin tensor
   - Create margin tensor: 0.7 if same address, else 0.5

3. **Hard address mining** (3 hours)
   - Add validation hook to track per-address error rates
   - Modify AddressAwarePKSampler to accept address weights
   - Curriculum: uniform epochs 1-10, weighted 11+

4. **Retrain** (48 hours)
   - 80 epochs at 384x384
   - Monitor worst-address accuracy every 2 epochs
   - Save checkpoints every 10 epochs

---

## Success Metrics

Track these during training:
- **Overall recall@1:** target >75% (from current unknown baseline)
- **Same-address accuracy:** target >70% for 2-tree addresses, >50% for 10+ tree addresses
- **Worst-10 addresses error rate:** reduce from 42% to <25% of total errors
- **Margin distribution:** shift from 92% <0.005 to >50% >0.01
- **Always-fail trees:** reduce 52 → <20

---

## Risk Mitigation

- **Memory overflow at 384x384:** Drop to P=10,K=4 or use gradient accumulation
- **Training instability with dynamic margins:** Start with uniform 0.6, gradually increase same-address to 0.7
- **Overfitting on hard addresses:** Monitor validation loss, add stronger dropout (0.3→0.4)
- **Slower convergence:** Extend warmup to 5 epochs, lower initial LR by 0.5x

---

## Timeline

- **Week 1:** Implement changes, test on small subset (10 addresses)
- **Week 2:** Full training run (80 epochs)
- **Week 3:** Error analysis, iterate on Tier 2 if needed

---

## Files to Modify

Priority 1 changes:
- `E:\tree_id_2.0\training\dataset_metric.py` (resolution)
- `E:\tree_id_2.0\training\losses_metric.py` (address-aware margins)
- `E:\tree_id_2.0\training\train_metric.py` (hard mining, batch size)
- `E:\tree_id_2.0\training\model_metric.py` (no changes)

Priority 2 changes:
- `E:\tree_id_2.0\training\losses_metric.py` (MS-Loss)
- `E:\tree_id_2.0\training\dataset_metric.py` (augmentation)
