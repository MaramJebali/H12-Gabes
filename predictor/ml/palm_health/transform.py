from torchvision import transforms


PALM_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.1909, 0.1937, 0.1896],
        std=[0.3242, 0.3258, 0.3336]
    )
])