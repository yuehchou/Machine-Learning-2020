import os
import re
import sys
import time
import torch
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from glob import glob
from PIL import Image
from model import decode8
from model import StudentNet
from training import loss_fn_kd
from training import run_epoch

workspace_dir = sys.argv[1]
teacher_model = './teacher_resnet18.bin'
save_model_path = 'student_model.bin'
save_8_bit_model_path = './8_bit_model.pkl'
batch_size = 128

class MyDataset(torch.utils.data.Dataset):

    def __init__(self, folderName, transform=None):
        self.transform = transform
        self.data = []
        self.label = []

        for img_path in sorted(glob(folderName + '/*.jpg')):
            try:
                # Get classIdx by parsing image path
                class_idx = int(re.findall(re.compile(r'\d+'), img_path)[1])
            except:
                # if inference mode (there's no answer), class_idx default 0
                class_idx = 0

            image = Image.open(img_path)
            # Get File Descriptor
            image_fp = image.fp
            image.load()
            # Close File Descriptor (or it'll reach OPEN_MAX)
            image_fp.close()

            self.data.append(image)
            self.label.append(class_idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = self.data[idx]
        if self.transform:
            image = self.transform(image)
        return image, self.label[idx]


trainTransform = transforms.Compose([
    transforms.RandomCrop(256, pad_if_needed=True, padding_mode='symmetric'),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
])

testTransform = transforms.Compose([
    transforms.CenterCrop(256),
    transforms.ToTensor(),
])

def get_dataloader(mode='training', batch_size=32):

    assert mode in ['training', 'testing', 'validation']

    dataset = MyDataset(
        '{}'.format(os.path.join(workspace_dir, mode)),
        transform=trainTransform if mode == 'training' else testTransform)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(mode == 'training'))

    return dataloader

# get dataloader
print("Reading data")
train_dataloader = get_dataloader('training', batch_size=32)
valid_dataloader = get_dataloader('validation', batch_size=32)

teacher_net = models.resnet18(pretrained=False, num_classes=11).cuda()
student_net = StudentNet(base=16).cuda()
teacher_net.load_state_dict(torch.load('{}'.format(teacher_model)))
optimizer = optim.AdamW(student_net.parameters(), lr=1e-3)

teacher_net.eval()
now_best_acc = 0
for epoch in range(200):
    student_net.train()
    train_loss, train_acc = run_epoch(
        train_dataloader,
        teacher_net,
        student_net,
        optimizer,
        update=True
    )
    student_net.eval()
    valid_loss, valid_acc = run_epoch(
        valid_dataloader,
        teacher_net,
        student_net,
        optimizer,
        update=False
    )

    if valid_acc > now_best_acc:
        now_best_acc = valid_acc
        torch.save(student_net.state_dict(), save_model_path)
    print('epoch {:>3d}: train loss: {:6.4f}, acc {:6.4f} valid loss: {:6.4f}, acc {:6.4f}'.format(
        epoch, train_loss, train_acc, valid_loss, valid_acc))

params = torch.load(save_model_path)
encode8(params, save_8_bit_model_path)
print("8-bit cost: {} bytes.".format(os.stat(save_8_bit_model_path).st_size))
