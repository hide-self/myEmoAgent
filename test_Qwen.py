# test_Qwen.py
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from PIL import Image
import torch

model_path = "/root/autodl-tmp/models/Qwen/Qwen2.5-VL-7B-Instruct"

# 使用 Qwen2_5_VL 专用类
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True   # 确保允许自定义代码
)
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

# 测试一张图片（请替换为实际存在的图片路径）
image = Image.open("./single_test/anger_00000.jpg")
prompt = "请用中文描述这张图片的内容，不超过20个字。"

messages = [
    {"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": prompt}
    ]}
]

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(
    text=[text],
    images=[image],
    padding=True,
    return_tensors="pt"
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=50)
response = processor.decode(outputs[0], skip_special_tokens=True)
print("模型回复：", response)
