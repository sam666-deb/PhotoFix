---
title: PhotoFix
emoji: 📷
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
short_description: Finds what's wrong with a photo and fixes it
---

# PhotoFix

Drop in a photo. PhotoFix detects exposure, contrast, color, noise and blur problems, fixes only what's
wrong, and checks its own result for over-editing before showing it to you.

- **Natural:** vivid, rule-based corrections driven by a neural defect analyzer.
- **Pro:** color and tone learned from a professional retoucher (MIT-Adobe FiveK, research use only).

This free demo runs on 2 CPUs, so photos are processed at up to 2048 px. Uploads are processed in memory and
never stored. For code, benchmarks and full-resolution local use, see
[github.com/sam666-deb/PhotoFix](https://github.com/sam666-deb/PhotoFix). Model weights:
[{MODEL_REPO}](https://huggingface.co/{MODEL_REPO}).
