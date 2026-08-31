import torch
from torch import nn
from torchvision import transforms, models
from PIL import Image

def load_classifier(model_path, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    model = models.resnet50(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 8)
    state_dict = torch.load(model_path, map_location=device)
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def predict_emotion(image_path, model, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    img = Image.open(image_path).convert('RGB')
    img_t = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_t)
        probs = torch.softmax(outputs, dim=1)
        pred_id = torch.argmax(probs, dim=1).item()
    # 需要 ID2EMO 映射，可以提前定义或从 info.json 读取
    ID2EMO = {0: "amusement", 1: "awe", 2: "contentment", 3: "excitement",
              4: "anger", 5: "disgust", 6: "fear", 7: "sadness"}
    return ID2EMO[pred_id], probs[0][pred_id].item()

# 使用示例
model = load_classifier("./weights/resnet50_emo8.pth")
emotion, confidence = predict_emotion("/root/autodl-tmp/EmoSet-118K/image/anger/anger_00000.jpg", model)
print(f"预测情感: {emotion}, 置信度: {confidence:.4f}")