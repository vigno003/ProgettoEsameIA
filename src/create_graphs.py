import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

def create_graphs(log_dir='runs/meteo_model'):
    writer = SummaryWriter(log_dir)
    writer.flush()
    writer.close()
    print(f"Graphs can be viewed using TensorBoard with the log directory: {log_dir}")