import torchvision
from torchvision  import transforms

train_data_path = r'./train/'

transforms = transforms.Compose([transforms.Resize(64),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[.485, .456, .406],
                                                     std=[.229,.224,.225])])

train_data = torchvision.datasets.ImageFolder(root = train_data_path, transform= transforms)