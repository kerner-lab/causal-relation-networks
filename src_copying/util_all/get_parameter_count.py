from model_all.model import Model


# note: assuming Transformers or Relation Networks
#     because these models only have fc layers
def get_parameter_count(module):
    out = 0
    for name, p in module.named_parameters():
        if "fc" in name:
            out += p.numel()
    print(f"Total number of parameters: {out / 1e6:.2f}M (excl. wte & wpe)")
    return out        

if __name__ == "__main__":
    model = Model()
    get_parameter_count(model)        
    model.pre_block.wte