import numpy as np
import torch
from torch import Tensor
from model import Generator
import matplotlib.pyplot as plt
from PIL import Image


device = "cuda" if torch.cuda.is_available() else "cpu"

gen_image_size = 28

model_path = "work_dirs/20240315-090721-mnist/latest_generator.pth"
generator = Generator(out_channels=1).to(device)
generator.load_state_dict(torch.load(model_path))
generator
