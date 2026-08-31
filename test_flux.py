#!/usr/bin/env python3
"""
测试 FLUX.2 [klein] 的图像编辑功能（图生图）
"""
import torch
from diffusers import Flux2KleinPipeline
from PIL import Image
import os

# -------------------- 配置 --------------------
MODEL_PATH = "/root/autodl-tmp/models/black-forest-labs/FLUX.2-klein-9B"
INPUT_IMAGE_PATH = "/root/autodl-tmp/EmoEdit-inference/Original_405/1.png"
OUTPUT_DIR = "./flux_edit_test"

# 编辑指令
EDIT_PROMPT = "Make the atmosphere exciting and vibrant, add fireworks in the sky, warm golden lighting, and cheerful colors."

# 图像尺寸
HEIGHT = 512
WIDTH = 512

# FLUX.2 推荐参数
GUIDANCE_SCALE = 1.0
NUM_INFERENCE_STEPS = 4
SEED = 42

# -------------------- 加载模型 --------------------
print(f"正在加载 FLUX.2 模型: {MODEL_PATH}")
pipe = Flux2KleinPipeline.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16
)
pipe.enable_model_cpu_offload()
print("模型加载完成。")

# -------------------- 加载输入图片 --------------------
if not os.path.exists(INPUT_IMAGE_PATH):
    print(f"警告: 未找到 {INPUT_IMAGE_PATH}，将生成一张测试渐变图。")
    input_image = Image.new('RGB', (WIDTH, HEIGHT), color=(73, 109, 137))
else:
    input_image = Image.open(INPUT_IMAGE_PATH).convert('RGB').resize((WIDTH, HEIGHT))

print("输入图片加载完成，开始执行情感编辑...")

# -------------------- 执行编辑（图生图） --------------------
generator = torch.Generator(device="cuda").manual_seed(SEED)

edited_image = pipe(
    prompt=EDIT_PROMPT,
    image=input_image,          # 【关键】传入原始图片作为参考
    height=HEIGHT,
    width=WIDTH,
    guidance_scale=GUIDANCE_SCALE,
    num_inference_steps=NUM_INFERENCE_STEPS,
    generator=generator,
).images[0]

# -------------------- 保存结果 --------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)
output_path = os.path.join(OUTPUT_DIR, "edited_output.png")
edited_image.save(output_path)
print(f"✅ 编辑成功！结果保存至: {output_path}")

# 同时保存原图方便对比
input_image.save(os.path.join(OUTPUT_DIR, "original_input.png"))
print("原图已保存用于对比。")

