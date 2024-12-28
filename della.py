import argparse
import torch
from torch import nn
import matplotlib.pyplot as plt
from stu import STU
import time
import random

save_folder = "lds_train_results2" 

uid = random.randint(1000,9999)

def compute_ar_x_preds(w: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    d_out, d_in, k = w.shape
    b, l, d_in_x = x.shape
    assert d_in == d_in_x, (
        f"Dimension mismatch: w.shape={w.shape}, x.shape={x.shape}"
    )

    o = torch.einsum("oik,bli->bklo", w, x)

    for i in range(k):
        o[:, i] = torch.roll(o[:, i], shifts=i, dims=1)

    m = torch.triu(torch.ones(k, l, dtype=o.dtype, device=o.device))  # [k, l]
    m = m.unsqueeze(-1).repeat(1, 1, d_out)  # [k, l, d_out]

    ar_x_preds = torch.sum(o * m, dim=1)  # now shape is [b, l, d_out]

    return ar_x_preds

class LDS(nn.Module):
    def __init__(self, state_dim, input_dim, output_dim, kx=10):
        super(LDS, self).__init__()
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kx = kx
        self.h0 = nn.Parameter(torch.randn(state_dim))
        init_A = torch.randn(state_dim)
        self.A = nn.Parameter(init_A / torch.max(torch.abs(init_A)))
        self.B = nn.Parameter(torch.randn(input_dim, state_dim) / input_dim)
        self.C = nn.Parameter(torch.randn(state_dim, output_dim) / state_dim)
        self.M = nn.Parameter(torch.randn(input_dim, output_dim, kx) / output_dim)

    def forward(self, inputs):
        device = inputs.device
        bsz, seq_len, _ = inputs.shape
        h_t = self.h0.expand(bsz, self.state_dim).to(device)
        A = self.A.flatten()
        all_h_t = []
        for t in range(seq_len):
            u_t = inputs[:, t, :]
            h_t = A * h_t + (u_t @ self.B)
            all_h_t.append(h_t.unsqueeze(1))
        all_h_t = torch.cat(all_h_t, dim=1)
        lds_out = torch.matmul(all_h_t, self.C)

        ar = compute_ar_x_preds(self.M, inputs)
        return lds_out + ar

    def compute_loss(self, inputs, targets):
        mse_loss = nn.MSELoss()
        outputs = self(inputs)
        return mse_loss(outputs, targets)



# Command-line argument parsing
parser = argparse.ArgumentParser(description="Train LDS model")
parser.add_argument("--layer_i", type=int, required=True, help="Layer index")
parser.add_argument("--state_dim", type=int, required=True, help="State dimension")
parser.add_argument("--batch_size", type=int, required=True, help="Batch size")
parser.add_argument("--epochs", type=int, required=True, help="Number of epochs")
parser.add_argument("--seq_len", type=int, required = True, help="Sequence length")
parser.add_argument("--kx", type=int, default = 10, required=False, help="direct input connections")
parser.add_argument("--lr", type=float, required=True, help="Learning rate")
args = parser.parse_args()

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the layer i weights
stu_layer_full = torch.load(f"./stu_layer_{args.layer_i}_500m_param_full.pt")
stu_layer_full.to(device)

# Initialize LDS model
lds = LDS(args.state_dim, 768, 768, args.kx).to(device)
optimizer = torch.optim.Adam(lds.parameters(), lr=args.lr)

# Training
lds_epochs = args.epochs
lds_loss_values = []

# Timer setup for saving the model every 20 minutes
last_save_time = time.time()
save_interval = 20 * 60  # 20 minutes in seconds

for epoch in range(lds_epochs):
    inputs = torch.randn(args.batch_size, args.seq_len, 768).to(device).to(torch.bfloat16)
    stu_outputs = stu_layer_full(inputs).to(device)

    optimizer.zero_grad()
    loss = lds.compute_loss(inputs.to(stu_outputs.dtype), stu_outputs)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(lds.parameters(), max_norm=1)
    lds_loss_values.append(loss.item())
    optimizer.step()

    with torch.no_grad():
        lds.A.data.clamp_(max=1, min=-1)

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

    # Save the model every 20 minutes
    if time.time() - last_save_time >= save_interval:
        torch.save({
            "lds_state_dict": lds.state_dict(),
            "lds_optimizer_state_dict": optimizer.state_dict(),
        }, f"./{save_folder}/{uid}_{args.layer_i}_{args.state_dim}_interim_lds_model_and_optimizer.pt")
        print(f"Model saved at epoch {epoch}.")
        last_save_time = time.time()

        with open(f"./{save_folder}/{uid}_interim_loss_progression.txt", "w") as f:
            for loss_value in lds_loss_values:
                f.write(f"{loss_value}\n")
        print(f"Loss progression saved at epoch {epoch}.")

# Save the loss progression to a file
with open(f"./{save_folder}/{uid}_loss_progression.txt", "w") as f:
    for loss_value in lds_loss_values:
        f.write(f"{loss_value}\n")

# Plot the loss progression
plt.figure()
plt.plot(range(len(lds_loss_values)), lds_loss_values, label="Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Progression")
plt.legend()
plt.savefig(f"./{save_folder}/{uid}_loss_progression.png")
plt.close()

# Save the LDS model and optimizer state
torch.save({
    "lds_state_dict": lds.state_dict(),
    "lds_optimizer_state_dict": optimizer.state_dict(),
    }, f"./{save_folder}/{uid}_{args.layer_i}_{args.state_dim}_{lds_loss_values[-1]:.5f}_lds_model_and_optimizer.pt")

print("Training complete. Files saved.")