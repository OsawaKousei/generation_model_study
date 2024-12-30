import matplotlib.pyplot as plt
import torch
from model import Diffuser, UNetCond

MODEL_PATH = "src/classifier_free_guidance/results/model/model_10.pth"
NUM_LABELS = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def show_img(images, num_rows=2, num_cols=10):
    fig = plt.figure(figsize=(num_cols, num_rows))
    i = 0
    for _ in range(num_rows):
        for _ in range(num_cols):
            fig.add_subplot(num_rows, num_cols, i + 1)
            plt.imshow(images[i], cmap="gray")
            plt.axis("off")
            i += 1
    plt.show()


net = UNetCond(num_labels=NUM_LABELS).to(device)
net.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))

labels = torch.randint(0, NUM_LABELS, (20,), device=device)
print(labels)

diffuser = Diffuser(device)
images = diffuser.sample(net, labels=labels)
show_img(images)
