import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

def create_graphs(log_dir='runs/meteo_model'):
    writer = SummaryWriter(log_dir)
    # Assuming you have already logged the data during training
    # You can use TensorBoard to visualize the graphs
    writer.flush()
    writer.close()
    print(f"Graphs can be viewed using TensorBoard with the log directory: {log_dir}")