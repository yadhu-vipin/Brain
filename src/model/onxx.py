import torch
import torchvision.models as models
print("Is CUDA available?", torch.cuda.is_available())


# Load your model
model = models.resnet50(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, len(["no_tumor", "glioma", "meningioma", "pituitary"]))  # Set classes
model.load_state_dict(torch.load("src/model/model.pth", map_location=torch.device('cpu')))
model.to(torch.device('cpu'))  # Explicitly map to CPU
model.eval()

dummy_input = torch.randn(1, 3, 224, 224, device=torch.device('cpu'))

torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    opset_version=11
)
print("Model exported successfully to model.onnx")
