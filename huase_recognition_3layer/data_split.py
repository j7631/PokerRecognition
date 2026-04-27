import os
import random
import shutil

# ================== 路径配置 ==================
INPUT_ROOT = "/Users/bytedance/Desktop/Davis/data"
OUTPUT_ROOT = "/Users/bytedance/Desktop/Davis/suit_dataset"

CLASSES = ["fangpian", "hongtao", "heitao", "meihua", "empty"]
TRAIN_RATIO = 0.8
IMG_EXT = (".jpg", ".png")

random.seed(42)

# ================== 工具函数 ==================
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def collect_images(class_name):
    """
    收集某一类的所有图片路径
    """
    paths = []

    class_path = os.path.join(INPUT_ROOT, class_name)

    # empty 没有子文件夹
    if class_name == "empty":
        for f in os.listdir(class_path):
            if f.lower().endswith(IMG_EXT):
                paths.append(os.path.join(class_path, f))
    else:
        # 花色下面有 1~10
        for rank in os.listdir(class_path):
            rank_path = os.path.join(class_path, rank)
            if not os.path.isdir(rank_path):
                continue
            for f in os.listdir(rank_path):
                if f.lower().endswith(IMG_EXT):
                    paths.append(os.path.join(rank_path, f))

    return paths

# ================== 创建目录 ==================
for split in ["train", "test"]:
    for cls in CLASSES:
        ensure_dir(os.path.join(OUTPUT_ROOT, split, cls))

# ================== 收集所有类别 ==================
all_class_images = {}
min_count = float("inf")

for cls in CLASSES:
    imgs = collect_images(cls)
    random.shuffle(imgs)
    all_class_images[cls] = imgs
    min_count = min(min_count, len(imgs))
    print(f"{cls}: {len(imgs)} images")

print(f"\n⚖️ 使用每类最小数量对齐: {min_count}\n")

# ================== 划分并复制 ==================
for cls, imgs in all_class_images.items():
    imgs = imgs[:min_count]  # 数量对齐

    split_idx = int(len(imgs) * TRAIN_RATIO)
    train_imgs = imgs[:split_idx]
    test_imgs = imgs[split_idx:]

    for p in train_imgs:
        name = os.path.basename(p)
        shutil.copy(
            p,
            os.path.join(OUTPUT_ROOT, "train", cls, name)
        )

    for p in test_imgs:
        name = os.path.basename(p)
        shutil.copy(
            p,
            os.path.join(OUTPUT_ROOT, "test", cls, name)
        )

    print(
        f"{cls}: train={len(train_imgs)}, test={len(test_imgs)}"
    )

print("\n✅ 花色 5 分类数据集构建完成")
