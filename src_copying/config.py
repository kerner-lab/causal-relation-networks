import sys

# note: Transformer, Linear Transformer
#       Relation Network, Linear Relation Network, Relation Network Ablation
model_name = "Linear Transformer"
string_size = int(sys.argv[1])
wandb_project_name = "proj240519_timing"
ablation_parameters = {"PreNorm": "exact",  # "exact", "approx", "no"
                       "PostNorm": "yes",  # "yes", "no"
                       "Activation": "exp"}  # "exp", "elu", "relu", "gelu"

# ----- #
# training logistics
# ----- #
if model_name == "Relation Network Ablation":
    my_run_name = "RN(A)-{}-{}-{}-{}".format(string_size,
                                             ablation_parameters["PreNorm"],
                                             ablation_parameters["PostNorm"],
                                             ablation_parameters["Activation"])
else:
    my_run_name = "{}-{}".format(model_name, string_size)
run_folder = "./{}".format(my_run_name)


# ----- #
# other
# ----- #
# dataset
vocab_size = 29
context_window = 2 * (string_size + 1)
num_class = vocab_size
# dataloader
num_worker_dataloader = 8  # note: use 0 to disable
# model
enable_causal_mode = True
emb_size = 192
if model_name == "Transformer":
    hid_size = 192 * 4
elif model_name == "Linear Transformer":
    hid_size = 192 * 4
elif model_name == "Relation Network":
    hid_size = 192
elif model_name == "Linear Relation Network":
    hid_size = 192
elif model_name == "Relation Network Ablation":
    hid_size = 192
else:
    raise Exception()
num_block = 12
num_head = 1
if model_name == "Transformer":
    if (emb_size % num_head) != 0:
        raise Exception()
    head_size = emb_size // num_head
elif model_name == "Linear Transformer":
    if (emb_size % num_head) != 0:
        raise Exception()
    head_size = emb_size // num_head
elif model_name == "Relation Network":
    if (hid_size % num_head) != 0:
        raise Exception()
    head_size = hid_size // num_head
elif model_name == "Linear Relation Network":
    if (hid_size % num_head) != 0:
        raise Exception()
    head_size = hid_size // num_head
elif model_name == "Relation Network Ablation":
    if (hid_size % num_head) != 0:
        raise Exception()
    head_size = hid_size // num_head
else:
    raise Exception()
# note: "Layer Normalization" or "RMS Normalization"
normalization_type = "Layer Normalization"
normalization_eps = 0.0
enable_relation_network_bias_1 = True
enable_relation_network_bias_2 = False
enable_relation_network_bias_3 = True
# optimization
enable_grad_clip = True
grad_clip_max_norm = 1.0
adamw_beta_1 = 0.9
adamw_beta_2 = 0.999
adamw_weight_decay = 0.0
adamw_eps = 1e-8
# learning rate schedule
# note: (lr_max, lr_terminal, idx_end_warm_up, idx_end_decay)
lr_schedule_param = (5e-4, 5e-4, 50, 1000)
# training
num_epoch = 1
# note: effective_batch_size = num_grad_accu * batch_size
if model_name in ["Relation Network", "Relation Network Ablation"]:
    num_grad_accu = 1
    batch_size = 320
else:
    num_grad_accu = 1
    batch_size = 320
# other
num_iter_train = 2000
num_iter_eval = 1
# general
wandb_run_name = my_run_name
random_seed = 12345
enable_torch_compile = True
print("# ----- #")
print("# Messages from config.py")
print("# ----- #")
print("string_size is", string_size)
print("context_window is", context_window)
print("model shape is ({}, {}, {})".format(emb_size, hid_size, num_block))
print("normalization type is", normalization_type)
print("learning rate schedule: ", lr_schedule_param)
print("effective batch size is ", num_grad_accu * batch_size)
print("run_name is", wandb_run_name)
print("# ----- #")
