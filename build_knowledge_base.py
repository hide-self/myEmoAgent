#!/usr/bin/env python3
"""
build_emotion_factor_tree.py
使用 Qwen2.5-VL (transformers 方式) 构建情感因子树
参考 EmoEdit 方法: 聚类 -> VLM摘要 -> 生成JSON树
"""

import os
import json
import random
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import open_clip
import re

# ==================== 配置 ====================
EMOSET_ROOT = "/root/autodl-tmp/EmoSet-118K/image"  # EmoSet 数据集根目录
OUTPUT_DIR = "./emotion_factor_tree"          # 输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 8种情感类别
EMOTIONS = ["amusement", "awe", "contentment", "excitement",
            "anger", "disgust", "fear", "sadness"]

# 聚类参数
CLUSTER_NUM = 5          # 每种情感期望的聚类数量（可根据实际情况调整）
SAMPLE_SIZE = 50         # 每种情感采样图片数（用于聚类）
DESCRIBE_SIZE = 10       # 每个聚类用于生成摘要的图片数

# 设备
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==================== 1. 加载CLIP模型（用于聚类） ====================
def load_clip_model():
    # 使用本地权重文件路径（绝对路径）
    local_weight_path = "/root/autodl-tmp/models/clip/vit-b-32/open_clip_model.safetensors"
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32",
        pretrained=local_weight_path
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    return model.to(DEVICE), tokenizer, preprocess

def get_clip_embedding(model, preprocess, image_path):
    """提取单张图片的CLIP embedding"""
    image = preprocess(Image.open(image_path).convert('RGB')).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        embedding = model.encode_image(image)
        embedding = normalize(embedding.cpu().numpy())
    return embedding.flatten()

# ==================== 2. 加载Qwen-VL模型（用于生成摘要） ====================
def load_qwen_model():
    """加载Qwen2.5-VL模型和处理器"""
    model_name = "/root/autodl-tmp/models/Qwen/Qwen2.5-VL-7B-Instruct"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True  # 建议加上，确保自定义代码可以执行
    )
    processor = AutoProcessor.from_pretrained(model_name)
    return model, processor


def describe_image_with_qwen(model, processor, image_path, prompt):
    """使用Qwen-VL描述图片"""
    image = Image.open(image_path).convert('RGB')
    conversation = [
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt}
        ]}
    ]
    text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(
        text=[text_prompt],
        images=[image],
        padding=True,
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.3,
            do_sample=True,
        )
    full_output = processor.decode(outputs[0], skip_special_tokens=True)
    if "assistant" in full_output:
        response = full_output.split("assistant")[-1].strip()
    else:
        response = full_output.strip()
    return response


def summarize_cluster_with_qwen(model, processor, descriptions, emotion):
    """使用Qwen文本模型总结聚类共性"""
    prompt = f"""
    以下是情感类别为 "{emotion}" 的一组图片的视觉描述（每一条对应一张图片）：
    {'; '.join(descriptions)}

    请分析这些描述，总结这些图片的 **共同视觉元素**（如颜色、物体、场景、光影、构图等），
    并给出 3~5 条具体的 **图像编辑建议**，用于强化 "{emotion}" 这种情感。

    输出格式为 JSON，包含两个字段：
    {{
        "common_elements": ["元素1", "元素2", ...],
        "editing_suggestions": ["建议1", "建议2", ...]
    }}
    只输出 JSON，不要有其他内容。
    """
    conversation = [
        {"role": "user", "content": [{"type": "text", "text": prompt}]}
    ]
    text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(
        text=[text_prompt],
        padding=True,
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.3,
            do_sample=True,
        )
    full_output = processor.decode(outputs[0], skip_special_tokens=True)
    if "assistant" in full_output:
        response = full_output.split("assistant")[-1].strip()
    else:
        response = full_output.strip()

    # 提取 JSON 对象（容错）
    json_match = re.search(r'\{.*\}', response, re.DOTALL)
    if json_match:
        return json_match.group()
    else:
        # 如果提取失败，返回原字符串，让上层 json.loads 捕获异常
        return response

# ==================== 3. 构建情感因子树 ====================
def build_emotion_factor_tree():
    # 加载模型
    print("🚀 加载 CLIP 模型...")
    clip_model, _, clip_preprocess = load_clip_model()
    print("🚀 加载 Qwen-VL 模型...")
    qwen_model, qwen_processor = load_qwen_model()

    all_trees = {}

    for emotion in EMOTIONS:
        print(f"\n📁 处理情感类别: {emotion}")
        emotion_dir = os.path.join(EMOSET_ROOT, emotion)
        if not os.path.isdir(emotion_dir):
            print(f"⚠️ 目录不存在: {emotion_dir}，跳过")
            continue

        # 获取所有图片
        all_images = [f for f in os.listdir(emotion_dir)
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not all_images:
            print(f"⚠️ 情感 '{emotion}' 下没有图片，跳过")
            continue

        # 采样图片用于聚类
        sample_files = random.sample(all_images, min(SAMPLE_SIZE, len(all_images)))
        print(f"   采样 {len(sample_files)} 张图片用于聚类")

        # 提取CLIP embeddings
        embeddings = []
        valid_files = []
        for img_file in tqdm(sample_files, desc="   提取CLIP特征"):
            img_path = os.path.join(emotion_dir, img_file)
            try:
                emb = get_clip_embedding(clip_model, clip_preprocess, img_path)
                embeddings.append(emb)
                valid_files.append(img_file)
            except Exception as e:
                print(f"   处理 {img_file} 失败: {e}")
                continue

        if len(embeddings) < CLUSTER_NUM:
            print(f"⚠️ 有效图片不足 {CLUSTER_NUM} 张，跳过")
            continue

        # K-Means聚类
        embeddings = np.array(embeddings)
        kmeans = KMeans(n_clusters=min(CLUSTER_NUM, len(embeddings)), random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)

        # 对每个聚类生成摘要
        clusters = {}
        for idx, label in enumerate(labels):
            clusters.setdefault(label, []).append(valid_files[idx])

        tree = {"emotion": emotion, "factors": []}

        for cluster_id, cluster_files in clusters.items():
            print(f"   处理聚类 {cluster_id+1}/{len(clusters)} (共 {len(cluster_files)} 张图片)")

            # 从聚类中采样图片用于描述
            desc_files = random.sample(cluster_files, min(DESCRIBE_SIZE, len(cluster_files)))
            descriptions = []

            for img_file in tqdm(desc_files, desc=f"      描述图片", leave=False):
                img_path = os.path.join(emotion_dir, img_file)
                prompt = "请用中文简洁描述这张图片的主要视觉元素、场景、颜色和氛围，不超过20个字。"
                try:
                    desc = describe_image_with_qwen(qwen_model, qwen_processor, img_path, prompt)
                    descriptions.append(desc)
                except Exception as e:
                    print(f"      描述 {img_file} 失败: {e}")
                    continue

            if not descriptions:
                continue

            # 总结聚类共性
            try:
                summary_json = summarize_cluster_with_qwen(qwen_model, qwen_processor, descriptions, emotion)
                summary = json.loads(summary_json)
                tree["factors"].append({
                    "cluster_id": int(cluster_id),
                    "image_count": len(cluster_files),
                    "summary": summary
                })
            except Exception as e:
                print(f"      总结聚类失败: {e}")
                continue

        all_trees[emotion] = tree
        print(f"   ✅ {emotion} 因子树构建完成，共 {len(tree['factors'])} 个因子")

    # 保存所有因子树
    output_path = os.path.join(OUTPUT_DIR, "emotion_factor_tree.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_trees, f, ensure_ascii=False, indent=2)
    print(f"\n🎉 所有情感因子树已保存至: {output_path}")

    # 同时保存为更简洁的格式（类似EmoEdit的因子树结构）
    simple_tree = {}
    for emotion, tree in all_trees.items():
        simple_tree[emotion] = {
            "common_elements": [],
            "editing_suggestions": []
        }
        for factor in tree["factors"]:
            simple_tree[emotion]["common_elements"].extend(factor["summary"].get("common_elements", []))
            simple_tree[emotion]["editing_suggestions"].extend(factor["summary"].get("editing_suggestions", []))

    simple_output_path = os.path.join(OUTPUT_DIR, "emotion_knowledge_base.json")
    with open(simple_output_path, 'w', encoding='utf-8') as f:
        json.dump(simple_tree, f, ensure_ascii=False, indent=2)
    print(f"🎉 简化版知识库已保存至: {simple_output_path}")

if __name__ == "__main__":
    build_emotion_factor_tree()

