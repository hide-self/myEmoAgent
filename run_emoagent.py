#!/usr/bin/env python3
"""
run_emoagent.py
EmoAgent 复现脚本（使用 FLUX.2 作为 Editing Agent）

- 使用 Qwen2.5-VL-7B 作为 Planning Agent 和 Critic Agent（推理）
- 使用 FLUX.2 [klein] 作为 Editing Agent（图生图）
- 使用 ResNet-50（训练于 EmoSet）作为情感分类器
- 情感知识库来自 build_knowledge_base.py 输出的 emotion_knowledge_base.json

模型分配：
  - GPU 0: Qwen2.5-VL-7B + ResNet-50
  - GPU 1: FLUX.2 [klein]

用法示例：
    python run_emoagent.py --input_dir /root/autodl-tmp/EmoEdit-inference/Original_405 --output_dir ./output_flux --num_plans 3
"""

import os
import sys
import json
import random
import argparse
from PIL import Image
import torch
import numpy as np
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from diffusers import Flux2KleinPipeline
import torch.nn.functional as F

# -------------------- 配置 --------------------
EMOTIONS = ["amusement", "awe", "contentment", "excitement",
            "anger", "disgust", "fear", "sadness"]
EMO2ID = {e: i for i, e in enumerate(EMOTIONS)}
ID2EMO = {i: e for e, i in EMO2ID.items()}

# 模型路径
QWEN_PATH = "/root/autodl-tmp/models/Qwen/Qwen2.5-VL-7B-Instruct"
FLUX_PATH = "/root/autodl-tmp/models/black-forest-labs/FLUX.2-klein-9B"
RESNET_WEIGHT = "./weights/resnet50_emo8.pth"
KNOWLEDGE_PATH = "./emotion_factor_tree/emotion_knowledge_base.json"

# 设备分配
DEVICE_QWEN = torch.device("cuda:0")
DEVICE_CLS = torch.device("cuda:0")   # 分类器与 Qwen 同卡
FLUX_GPU_ID = 1                       # FLUX 使用 GPU1

# FLUX 生成参数
FLUX_HEIGHT = 512
FLUX_WIDTH = 512
FLUX_GUIDANCE_SCALE = 1.0
FLUX_NUM_STEPS = 4
FLUX_SEED = 42

# -------------------- 加载分类器 --------------------
def load_classifier(weight_path, device):
    import torchvision.models as models
    model = models.resnet50(weights=None)
    num_features = model.fc.in_features
    model.fc = torch.nn.Linear(num_features, 8)
    state_dict = torch.load(weight_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model

# -------------------- 加载 Qwen (GPU0) --------------------
def load_qwen():
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        QWEN_PATH,
        torch_dtype=torch.bfloat16,
        device_map={"": DEVICE_QWEN},
        trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(QWEN_PATH, trust_remote_code=True)
    return model, processor

# -------------------- 加载 FLUX (使用 CPU offload 到 GPU1) --------------------
def load_flux_editor():
    print(f"加载 FLUX.2 模型 (CPU offload 到 GPU{FLUX_GPU_ID})...")
    pipe = Flux2KleinPipeline.from_pretrained(
        FLUX_PATH,
        torch_dtype=torch.bfloat16
    )
    # 启用 CPU offload，指定 GPU ID
    pipe.enable_model_cpu_offload(gpu_id=FLUX_GPU_ID)
    print("FLUX.2 模型就绪 (offload 模式)")
    return pipe

# -------------------- 情感知识库 --------------------
class KnowledgeBase:
    def __init__(self, json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

    def get_suggestions(self, emotion):
        return self.data.get(emotion, {}).get("editing_suggestions", [])

# -------------------- 日志记录工具（同时输出到终端和文件） --------------------
class Tee:
    def __init__(self, filename):
        self.file = open(filename, 'w')
        self.stdout = sys.stdout

    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)
        self.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        self.file.close()

# -------------------- EmoAgent --------------------
class EmoAgent:
    def __init__(self, qwen_model, qwen_processor, flux_pipe, classifier, knowledge_base):
        self.qwen = qwen_model
        self.qwen_processor = qwen_processor
        self.flux_pipe = flux_pipe
        self.classifier = classifier
        self.kb = knowledge_base

    def get_image_emotion(self, image):
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        img_t = transform(image).unsqueeze(0).to(DEVICE_CLS)
        with torch.no_grad():
            logits = self.classifier(img_t)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        pred_id = np.argmax(probs)
        return pred_id, probs

    def qwen_chat(self, image, prompt, max_new_tokens=512):
        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
            ]}
        ]
        text = self.qwen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.qwen_processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt"
        ).to(self.qwen.device)
        with torch.no_grad():
            outputs = self.qwen.generate(**inputs, max_new_tokens=max_new_tokens, temperature=0.7)
        response = self.qwen_processor.decode(outputs[0], skip_special_tokens=True)
        if "assistant" in response:
            response = response.split("assistant")[-1].strip()
        return response

    def plan_edits(self, image, target_emotion, num_plans=3):
        desc_prompt = "请用中文简洁描述这张图片的主要视觉元素、场景、颜色和氛围，不超过30个字。"
        img_desc = self.qwen_chat(image, desc_prompt, max_new_tokens=50)
        print(f"  图片描述: {img_desc}")

        suggestions = self.kb.get_suggestions(target_emotion)
        if not suggestions:
            suggestions = ["调整颜色基调更符合情感", "添加与情感相关的物体", "改变背景氛围"]
        random.shuffle(suggestions)

        plan_prompt = f"""
        你是一个图像编辑规划专家。给定一张图片，其内容描述为："{img_desc}"。
        目标情感是："{target_emotion}"。
        以下是一些可用的编辑元素参考（从情感知识库中提取）：
        {', '.join(suggestions[:5])}

        请生成 {num_plans} 个不同的编辑计划，每个计划包含 1~3 条具体的编辑指令（用自然语言描述，如“将天空改为日落”）。
        指令要清晰、可执行，适合用于图像编辑模型。
        输出格式为 JSON 列表，每个元素是一个计划（计划本身是一个指令列表）。
        只输出 JSON，不要其他内容。
        """
        response = self.qwen_chat(image, plan_prompt, max_new_tokens=512)
        try:
            import re
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                plans = json.loads(json_match.group())
            else:
                plans = json.loads(response)
        except:
            plans = [[random.choice(suggestions)] for _ in range(num_plans)]

        # 规范化每个计划：确保最终是一个字符串列表
        def normalize_plan(plan):
            if isinstance(plan, dict):
                # 提取字典中所有值（可能是字符串或列表）并扁平化
                items = []
                for v in plan.values():
                    if isinstance(v, list):
                        items.extend([str(x) for x in v if x])
                    else:
                        items.append(str(v))
                return items
            elif isinstance(plan, str):
                return [plan]
            elif isinstance(plan, list):
                # 确保每个元素是字符串
                return [str(x) for x in plan if x]
            else:
                return [str(plan)]

        plans = [normalize_plan(p) for p in plans]
        plans = plans[:num_plans]
        return plans

    def execute_plan(self, image, instructions):
        """使用 FLUX 按顺序执行每条指令（图生图）"""
        current_img = image.resize((FLUX_WIDTH, FLUX_HEIGHT))
        for ins in instructions:
            print(f"    执行指令: {ins}")
            generator = torch.Generator(device=f"cuda:{FLUX_GPU_ID}").manual_seed(FLUX_SEED)
            # FLUX 图生图调用
            edited = self.flux_pipe(
                prompt=ins,
                image=current_img,
                height=FLUX_HEIGHT,
                width=FLUX_WIDTH,
                guidance_scale=FLUX_GUIDANCE_SCALE,
                num_inference_steps=FLUX_NUM_STEPS,
                generator=generator,
            ).images[0]
            current_img = edited
        return current_img

    def critique(self, image, target_emotion, original_image=None):
        pred_id, probs = self.get_image_emotion(image)
        target_id = EMO2ID[target_emotion]
        is_correct = (pred_id == target_id)
        if is_correct:
            return True, None
        orig_desc = ""
        if original_image is not None:
            desc_prompt = "描述这张图片的主要视觉元素、场景、颜色和氛围，不超过30个字。"
            orig_desc = self.qwen_chat(original_image, desc_prompt, max_new_tokens=50)
        critique_prompt = f"""
        你是一个情感图像编辑批评家。
        原始图片描述：{orig_desc}
        目标情感：{target_emotion}
        当前编辑后的图片情感预测为：{ID2EMO[pred_id]}（概率{probs[pred_id]:.2f}），但目标应为 {target_emotion}。
        请分析当前图片与目标情感之间的差距，并提出一条具体的编辑指令（仅一条）来修正，使图像更贴近目标情感。
        只输出指令，不要其他内容。
        """
        suggestion = self.qwen_chat(image, critique_prompt, max_new_tokens=100)
        return False, suggestion

    def process(self, image, target_emotion, num_plans=3, max_iter=2):
        original = image.copy()
        plans = self.plan_edits(image, target_emotion, num_plans)
        final_results = []
        for plan_idx, instructions in enumerate(plans):
            print(f"  执行计划 {plan_idx+1}/{len(plans)}: {instructions}")
            current_img = image.copy()
            current_img = self.execute_plan(current_img, instructions)
            for it in range(max_iter):
                correct, suggestion = self.critique(current_img, target_emotion, original)
                if correct:
                    print(f"    计划 {plan_idx+1} 在第 {it+1} 次迭代后情感正确")
                    break
                else:
                    print(f"    计划 {plan_idx+1} 情感不正确，迭代 {it+1}: 建议修改 '{suggestion}'")
                    current_img = self.execute_plan(current_img, [suggestion])
            else:
                print(f"    计划 {plan_idx+1} 达到最大迭代次数")
            final_results.append(current_img)
        return final_results

# -------------------- 主函数 --------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="/root/autodl/EmoEdit-inference/Original_405")
    parser.add_argument("--output_dir", type=str, default="./output_flux")
    parser.add_argument("--num_plans", type=int, default=3)
    parser.add_argument("--max_iter", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    # 设置日志：同时输出到终端和文件（覆盖写入）
    log_file = os.path.join(args.output_dir, "run_log.txt")
    sys.stdout = Tee(log_file)
    print("="*50)
    print("运行日志将同时保存到:", log_file)

    print("加载模型...")
    print("加载 Qwen2.5-VL 到 GPU0...")
    qwen_model, qwen_processor = load_qwen()
    print("加载情感分类器 ResNet-50 到 GPU0...")
    classifier = load_classifier(RESNET_WEIGHT, DEVICE_CLS)
    print("加载 FLUX.2 编辑模型 (offload 到 GPU1)...")
    flux_pipe = load_flux_editor()
    print("加载情感知识库...")
    kb = KnowledgeBase(KNOWLEDGE_PATH)
    agent = EmoAgent(qwen_model, qwen_processor, flux_pipe, classifier, kb)

    image_files = [f for f in os.listdir(args.input_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))]
    print(f"找到 {len(image_files)} 张图片")

    emo_a_correct = 0
    emo_s_sum = 0.0
    total = 0

    for img_file in tqdm(image_files, desc="处理图片"):
        img_path = os.path.join(args.input_dir, img_file)
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            continue
        target_emotion = random.choice(EMOTIONS)
        print(f"\n处理 {img_file} -> 目标情感: {target_emotion}")

        _, orig_probs = agent.get_image_emotion(image)
        target_id = EMO2ID[target_emotion]
        orig_target_prob = orig_probs[target_id]

        edited_images = agent.process(image, target_emotion, args.num_plans, args.max_iter)

        for idx, edited_img in enumerate(edited_images):
            out_name = f"{os.path.splitext(img_file)[0]}_{target_emotion}_plan{idx+1}.png"
            out_path = os.path.join(args.output_dir, out_name)
            edited_img.save(out_path)

            pred_id, probs = agent.get_image_emotion(edited_img)
            is_correct = (pred_id == target_id)
            emo_a_correct += int(is_correct)
            target_prob = probs[target_id]
            emo_s_sum += target_prob - orig_target_prob
            total += 1

    emo_a = emo_a_correct / total if total else 0
    emo_s_avg = emo_s_sum / total if total else 0
    print("\n" + "="*50)
    print(f"定量评估结果（共 {total} 个编辑结果）")
    print(f"Emo-A (情感准确率): {emo_a*100:.2f}%")
    print(f"Emo-S (情感强度平均增量): {emo_s_avg:.4f}")
    print(f"编辑结果保存于: {args.output_dir}")

    # 恢复标准输出（程序结束会自动关闭，但最好显式恢复，避免影响后续）
    sys.stdout.flush()
    if hasattr(sys.stdout, 'close'):
        sys.stdout.close()
    sys.stdout = sys.__stdout__

if __name__ == "__main__":
    main()