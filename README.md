# Ranking-aware adapter for text-driven image ordering with CLIP
---
The official implementation of the paper.

**Ranking-aware Adapter** aims to adapt the pre-trained model for text-guided image ranking. By leveraging a special designed relational attention, we extract the text-conditioned visual distinction from image pairs as an additional supervision for boosting the ranking performance. The results demonstrate that this light-weighted adapter with the ranking-aware module enables a pre-trained CLIP model to support various image ranking tasks across domains, including object count sorting, image quality assessment, and facial age estimation.

[Ranking-aware adapter for text-driven image ordering with CLIP](https://arxiv.org/abs/2412.06760)

---
## Installation
The repository can be installed via
```
$ poetry install --with cuda
$ poetry run pip install libs/open_clip
```


## Run Inference
...

Here are some pretrained checkpoints.
1. [Object count sorting](...)
2. [Image quality assessment (MOS)](...)
3. [Multiple tasks (count, MOS, age, hci)](...)

## To do list
The repo is kept updating, stay tuned!
- [x] Inference & Demo code
- [ ] Upload checkpoints to public accessible cloud.
- [ ] Visualization examples and more details.
- [ ] Training code
